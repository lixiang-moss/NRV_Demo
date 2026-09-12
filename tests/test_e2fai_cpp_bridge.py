"""One opt-in, camera/GPU-free interoperability check for the C++ bridge.

NRV_TEST_CPP_BRIDGE=1 python -m unittest discover -s tests \
    -p 'test_e2fai_cpp_bridge.py' -v
"""
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def run_probe():
    import signal
    import socket
    import sys
    import tempfile
    import time
    import xmlrpc.client

    import numpy as np
    import rospy
    from dvs_msgs.msg import Event, EventArray
    from nrv_demo.msg import E2faiResult
    from std_msgs.msg import String

    sys.path.insert(0, str(ROOT / "ros_ws/src/nrv_demo/scripts"))
    from e2fai_protocol import EVENT_DTYPE, recv_packet, send_packet

    def wait_for(predicate, description, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        raise RuntimeError("Timed out: " + description)

    core = subprocess.Popen(["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process = None
    connection = None
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(10)
    with tempfile.TemporaryDirectory(prefix="cpp_bridge_interop_") as directory:
        output = Path(directory)
        log = (output / "bridge.log").open("w")
        try:
            def master_ready():
                try:
                    return xmlrpc.client.ServerProxy("http://127.0.0.1:11311").getPid("/probe")[0] == 1
                except Exception:
                    return False
            wait_for(master_ready, "isolated ROS master")
            rospy.init_node("cpp_bridge_probe", disable_signals=True)
            results, statuses = [], []
            publisher = rospy.Publisher("/dvs/events", EventArray, queue_size=10)
            result_sub = rospy.Subscriber("/nrv_e2fai/result", E2faiResult, results.append, queue_size=10)
            status_sub = rospy.Subscriber("/nrv_e2fai/status", String,
                                          lambda message: statuses.append(json.loads(message.data)), queue_size=10)
            process = subprocess.Popen([
                "rosrun", "nrv_demo", "e2fai_bridge", "_session_id:=interop-session",
                "_host:=127.0.0.1", "_port:=" + str(server.getsockname()[1]),
                "_output_dir:=" + directory, "_perf_enabled:=false"
            ], stdout=log, stderr=subprocess.STDOUT)
            connection, _ = server.accept()
            wait_for(lambda: publisher.get_num_connections() > 0 and result_sub.get_num_connections() > 0,
                     "ROS event and result subscribers")
            original = np.array([(959, 719, 1_800_000_000_000_000_019, 1),
                                 (2, 3, 1_800_000_000_000_000_020, 0),
                                 (4, 5, 1_800_000_000_000_000_023, 1)], dtype=EVENT_DTYPE)
            message = EventArray()
            message.width, message.height = 960, 720
            stamp = int(original["timestamp_ns"][-1])
            message.header.stamp = rospy.Time(stamp // 1_000_000_000, stamp % 1_000_000_000)
            for record in original:
                event = Event()
                event.x, event.y, event.polarity = int(record["x"]), int(record["y"]), bool(record["polarity"])
                timestamp = int(record["timestamp_ns"])
                event.ts = rospy.Time(timestamp // 1_000_000_000, timestamp % 1_000_000_000)
                message.events.append(event)
            publisher.publish(message)
            published_ros_sequence = int(message.header.seq)
            kind, metadata, payload = recv_packet(connection)
            assert kind == "events" and metadata["session_id"] == "interop-session", metadata
            assert metadata["batch_seq"] == 0 and metadata["count"] == 3, metadata
            assert metadata["ros_header_seq"] == published_ros_sequence and metadata["ros_sequence_valid"] is True, metadata
            assert metadata["header_stamp_ns"] == stamp, metadata
            assert metadata["width"] == 960 and metadata["height"] == 720, metadata
            assert bytes(payload) == original.tobytes(), "Event bytes or integer timestamps changed"

            gray = bytes([1, 2, 3, 4])
            color = bytes(range(12))
            flow = np.array([1.25, -2.5, 3.75, -4.0, 0.0, 5.5, 6.0, -7.0], dtype="<f4").tobytes()
            output_meta = dict(session_id="interop-session", window_id=7, source_batch_first=0,
                               source_batch_last=0, window_start_ns=int(original["timestamp_ns"][0]),
                               window_end_ns=int(original["timestamp_ns"][0]) + 100_000_000,
                               event_first_ns=int(original["timestamp_ns"][0]), event_last_ns=stamp,
                               event_count=3, width=2, height=2, reset_reason="ros_sequence_gap",
                               reset_count=1, source_callback_ns=metadata["callback_ns"],
                               completed_ns=time.monotonic_ns())
            send_packet(connection, "result", dict(output_meta, session_id="stale-session"), gray + color + flow)
            send_packet(connection, "result", output_meta, gray + color + flow)
            wait_for(lambda: len(results) == 1, "typed ROS result")
            result = results[0]
            for field, expected in output_meta.items():
                assert getattr(result, field) == expected, field
            assert result.header.stamp.to_nsec() == output_meta["window_end_ns"]
            assert bytes(result.gray.data) == gray and result.gray.encoding == "mono8"
            assert bytes(result.flow_preview.data) == color and result.flow_preview.encoding == "rgb8"
            assert bytes(result.flow.data) == flow and result.flow.encoding == "32FC2"
            assert not result.flow.is_bigendian and result.flow.step == 16

            send_packet(connection, 'status', dict(session_id='interop-session', state='catching_up',
                        processing_generation=1, reason='queue_freshness'))
            wait_for(lambda: any(s['state'] == 'catching_up' for s in statuses), 'catch-up status')
            # Old generation must be ignored even if its TCP packet is complete.
            send_packet(connection, 'result', output_meta, gray + color + flow)
            send_packet(connection, 'result', dict(output_meta, window_id=8, processing_generation=1), gray + color + flow)
            wait_for(lambda: len(results) >= 2, 'fresh-generation result')
            assert [r.window_id for r in results] == [7, 8], results

            send_packet(connection, "error", dict(session_id="interop-session", code="probe_stop", message="test pause"))
            wait_for(lambda: any(status["state"] == "paused" for status in statuses), "worker error pause")
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            assert process.returncode == 0, process.returncode
            summary = json.loads((output / "bridge_summary.json").read_text())
            assert summary["worker_errors"] == 1 and summary["local_errors"] == 0, summary
            assert summary["error_code"] == "probe_stop" and summary["stop_reason"] == "failure", summary
            assert summary["batches"] == 1 and summary["results"] == 2, summary
            assert not list(output.glob("performance_*.jsonl")), "Disabled measurement wrote a file"
            print("CPP_BRIDGE_INTEROP " + json.dumps({"status": "PASS", "original_events_exact": True,
                  "integer_metadata_exact": True, "result_payloads_exact": True,
                  "stale_session_ignored": True, "worker_error_paused": True, "summary": summary}), flush=True)

            # Hold connection establishment so the actual C++ unsent FIFO fills.
            # Tiny complete batches exercise the 32-batch guard without large allocations.
            connection.close()
            connection = None
            server.close()
            server = socket.socket()
            server.bind(('127.0.0.1', 0))
            server.settimeout(10)
            wait_for(lambda: publisher.get_num_connections() == 0, 'old bridge shutdown')
            process = subprocess.Popen([
                'rosrun', 'nrv_demo', 'e2fai_bridge', '_session_id:=overflow-session',
                '_host:=127.0.0.1', '_port:=' + str(server.getsockname()[1]),
                '_output_dir:=' + directory, '_perf_enabled:=false'
            ], stdout=log, stderr=subprocess.STDOUT)
            wait_for(lambda: publisher.get_num_connections() > 0, 'new event subscriber')
            for _ in range(34):
                publisher.publish(message)
                time.sleep(.02)
            wait_for(lambda: any(s['session_id'] == 'overflow-session' and s['batches'] == 34
                                 for s in statuses), 'sender FIFO overflow')
            overflow_status = next(s for s in reversed(statuses) if s['session_id'] == 'overflow-session')
            assert overflow_status['state'] == 'catching_up', overflow_status
            assert overflow_status['catchup_discarded_batches'] == 17, overflow_status
            server.listen(1)
            connection, _ = server.accept()
            retained = [recv_packet(connection) for _ in range(17)]
            assert [p[1]['batch_seq'] for p in retained] == list(range(17, 34)), retained
            assert all(p[1]['bridge_generation'] == 1 and bytes(p[2]) == original.tobytes()
                       for p in retained), 'Retained events changed or lost the recovery marker'
            send_packet(connection, 'result', dict(output_meta, session_id='overflow-session',
                        bridge_generation=0), gray + color + flow)
            send_packet(connection, 'result', dict(output_meta, session_id='overflow-session',
                        window_id=9, bridge_generation=1, processing_generation=1), gray + color + flow)
            wait_for(lambda: len(results) >= 3, 'result after upstream recovery')
            assert [r.window_id for r in results] == [7, 8, 9], results
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            recovered = json.loads((output / 'bridge_summary.json').read_text())
            assert recovered['local_errors'] == recovered['worker_errors'] == 0, recovered
            assert recovered['catchup_discarded_batches'] == 17 and recovered['results'] == 1, recovered
            print('CPP_BRIDGE_CATCHUP ' + json.dumps(recovered), flush=True)
            result_sub.unregister()
            status_sub.unregister()
        finally:
            if connection is not None:
                connection.close()
            server.close()
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
            print((output / "bridge.log").read_text(), flush=True)
            rospy.signal_shutdown("probe complete")
            core.terminate()
            try:
                core.wait(timeout=10)
            except subprocess.TimeoutExpired:
                core.kill()
                core.wait()


@unittest.skipUnless(os.environ.get("NRV_TEST_CPP_BRIDGE") == "1", "Set NRV_TEST_CPP_BRIDGE=1 for isolated C++ bridge check")
class CppBridgeInteropTests(unittest.TestCase):
    def test_nef1_event_and_result_interoperability_without_gpu_or_camera(self):
        result = subprocess.run([
            "docker", "run", "--rm", "--network", "none",
            "-e", "ROS_MASTER_URI=http://127.0.0.1:11311", "-e", "ROS_HOSTNAME=127.0.0.1",
            "-e", "NRV_CPP_BRIDGE_INSIDE=1", "-v", str(ROOT) + ":/workspace:ro",
            os.environ.get("NRV_ROS_TEST_IMAGE", "nrv-demo:noetic"),
            "python3", "/workspace/tests/test_e2fai_cpp_bridge.py"
        ], text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = next(line for line in result.stdout.splitlines() if line.startswith("CPP_BRIDGE_INTEROP "))
        self.assertEqual(json.loads(evidence.split(" ", 1)[1])["status"], "PASS")
        print(evidence, flush=True)
        print(next(line for line in result.stdout.splitlines() if line.startswith('CPP_BRIDGE_CATCHUP ')), flush=True)


if __name__ == "__main__":
    if os.environ.get("NRV_CPP_BRIDGE_INSIDE") == "1":
        run_probe()
    else:
        unittest.main()

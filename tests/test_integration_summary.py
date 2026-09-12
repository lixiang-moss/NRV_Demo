import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('check_run', Path(__file__).resolve().parents[1]/'scripts/check_run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IntegrationSummaryTests(unittest.TestCase):
    def test_catchup_is_reported_as_data_loss_not_a_session_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'summary.json').write_text('{"status":"PASS","raw_sequence_gaps":0}')
            (folder/'sessions/test').mkdir(parents=True)
            (folder/'sessions/test/bridge_summary.json').write_text('{"state":"stopped"}')
            (folder/'e2fai_session_001.json').write_text(
                '{"results":4,"error":null,"catchup_count":1,"catchup_discarded_batches":3}')
            result = module.inspect_run(folder)
            self.assertEqual(result['status'], 'DEGRADED')
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['catchup_discarded_batches'], 3)
            self.assertEqual(result['ros_sequence_gap_incidents'], 0)

    def test_capture_pass_does_not_hide_model_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'summary.json').write_text(json.dumps(dict(status='PASS',raw_sequence_gaps=2)))
            (folder/'sessions/test').mkdir(parents=True)
            (folder/'sessions/test/bridge_summary.json').write_text('{"state":"stopped","local_errors":0,"worker_errors":0}')
            self.assertEqual(module.inspect_run(folder)['status'],'DEGRADED')
            (folder/'e2fai_session_001.json').write_text(json.dumps(dict(results=4,error={'code':'queue_overflow'},reset_count=0)))
            self.assertEqual(module.inspect_run(folder)['status'],'DEGRADED')
            (folder/'e2fai_session_001.json').write_text(json.dumps(dict(results=4,error=None,reset_count=0)))
            self.assertEqual(module.inspect_run(folder)['status'],'PASS')
            (folder/'sessions/test/bridge_summary.json').write_text('{"state":"stopped","local_errors":1}')
            self.assertEqual(module.inspect_run(folder)['status'],'DEGRADED')
            (folder/'sessions/test/bridge_summary.json').write_text('{"state":"paused","detail":"bridge overflow"}')
            self.assertEqual(module.inspect_run(folder)['status'],'DEGRADED')
            self.assertEqual(json.loads((folder/'summary.json').read_text())['raw_sequence_gaps'],2)

    def test_stream_resets_and_disabled_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'summary.json').write_text('{"status":"PASS"}')
            (folder/'sessions/test').mkdir(parents=True)
            (folder/'sessions/test/bridge_summary.json').write_text('{"state":"stopped"}')
            (folder/'e2fai_session_001.json').write_text('{"results":4,"error":null,"reset_count":7}')
            self.assertEqual(module.inspect_run(folder)['status'],'PASS')
            (folder/'e2fai_session_001.json').write_text('{"results":4,"error":null,"reset_count":7,"ros_sequence_gap_incidents":1}')
            self.assertEqual(module.inspect_run(folder)['status'],'DEGRADED')
            self.assertEqual(module.inspect_run(folder,False)['status'],'PASS')


if __name__=='__main__': unittest.main()

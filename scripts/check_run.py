#!/usr/bin/env python3
"""Report RAW capture and E2FAI session results without the removed demo algorithm."""
import argparse
import json
from pathlib import Path


def inspect_run(folder, model_enabled=True):
    capture_path = folder / 'summary.json'
    capture = json.loads(capture_path.read_text()) if capture_path.exists() else None
    sessions = [json.loads(path.read_text()) for path in sorted(folder.glob('e2fai_session_*.json'))]
    bridges = [json.loads(path.read_text()) for path in sorted(folder.glob('sessions/*/bridge_summary.json'))]
    errors = [session['error'] for session in sessions if session.get('error')]
    bridge_errors = [bridge for bridge in bridges if bridge.get('state') == 'paused'
                     or bridge.get('local_errors', 0) or bridge.get('worker_errors', 0)]
    checks = dict(raw_capture_passed=bool(capture and capture.get('status') == 'PASS'))
    if model_enabled:
        checks.update(model_session_recorded=bool(sessions),
                      model_produced_results=any(s.get('results', 0) > 0 for s in sessions),
                      model_no_session_errors=not errors,
                      bridge_session_recorded=bool(bridges),
                      bridge_no_session_errors=not bridge_errors,
                      model_no_ros_sequence_gaps=not any(s.get('ros_sequence_gap_incidents', 0) for s in sessions))
    status = 'PASS' if all(checks.values()) else 'DEGRADED' if checks['raw_capture_passed'] else 'FAIL'
    result = dict(status=status, checks=checks, model_enabled=model_enabled,
                  capture_summary='summary.json', model_session_count=len(sessions),
                  model_results=sum(s.get('results', 0) for s in sessions),
                  stream_resets=sum(s.get('reset_count', 0) for s in sessions),
                  ros_sequence_gap_incidents=sum(s.get('ros_sequence_gap_incidents', 0) for s in sessions),
                  errors=errors, bridge_errors=bridge_errors,
                  note='RAW capture PASS does not establish model throughput or latency.')
    (folder / 'integration_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--model-disabled', action='store_true')
    args=p.parse_args()
    result=inspect_run(args.folder, not args.model_disabled)
    print(json.dumps(result, indent=2))
    print('Integration summary:', args.folder/'integration_summary.json')
    raise SystemExit(0 if result['status']=='PASS' else 1)

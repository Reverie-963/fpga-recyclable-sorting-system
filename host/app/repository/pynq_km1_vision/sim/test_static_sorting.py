"""Behavior tests for the offline static sorting contract."""
import copy
import json
from pathlib import Path
import unittest
from src.static_sorting import SortingSimulator, replay


class StaticSortingTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((Path(__file__).resolve().parents[1] / 'data/static_sorting/demo.json').read_text(encoding='utf-8'))
        self.config = self.doc['config']
        self.request = self.doc['requests'][0]['request']

    def run_case(self, reason, mutate_request=None, mutate_config=None, actions=0):
        if mutate_request:
            mutate_request(self.request)
        if mutate_config:
            mutate_config(self.config)
        result = SortingSimulator(self.config).process(self.request, 1010)
        self.assertEqual(result['reason'], reason)
        self.assertEqual(len(result['actions']), actions)
        return result

    def test_acceptance_replay(self):
        path = Path(__file__).resolve().parents[1] / 'data/static_sorting/acceptance.json'
        document = json.loads(path.read_text(encoding='utf-8'))
        for entry, record in zip(document['requests'], replay(document)):
            with self.subTest(name=entry['name']):
                self.assertEqual(record['status'], entry['expected_status'])
                self.assertEqual(record['reason'], entry['expected_reason'])
                if record['status'] == 'rejected':
                    self.assertEqual(record['actions'], [])

    def test_complete_and_replay(self):
        first = replay(self.doc)
        self.assertEqual(first, replay(copy.deepcopy(self.doc)))
        self.assertEqual(first[0]['status'], 'completed')
        self.assertEqual(first[0]['plan']['pick_xy_mm'], [20, 20])
        self.assertEqual(len(first[0]['actions']), 7)
        self.assertFalse(first[0]['automatic_grip_check'])
        self.assertFalse(first[0]['automatic_drop_detection'])
        self.assertNotIn('grip_verified', first[0]['states'])

    def test_all_classes(self):
        for label in self.config['bins']:
            self.request['instances'][0]['label'] = label
            out = SortingSimulator(self.config).process(self.request, 1010)
            self.assertEqual(out['plan']['bin_id'], label+'-bin')

    def test_empty(self):
        self.run_case('empty_scene', lambda r: r.update(instances=[]))

    def test_multiple_objects(self):
        self.run_case('ambiguous_scene', lambda r: r['instances'].append(copy.deepcopy(r['instances'][0])))

    def test_low_confidence(self):
        self.run_case('low_confidence', lambda r: r['instances'][0].update(confidence=0.1))

    def test_stale(self):
        self.run_case('stale_frame', lambda r: r.update(received_age_ms=501))

    def test_queue_delay(self):
        self.run_case('stale_frame', lambda r: r.update(received_at_ms=0))

    def test_future_timestamp(self):
        self.run_case('stale_frame', lambda r: r.update(received_at_ms=2000))

    def test_missing_calibration(self):
        self.run_case('missing_calibration', mutate_config=lambda c: c.pop('calibration'))

    def test_wrong_camera_calibration(self):
        self.run_case('missing_calibration', mutate_config=lambda c: c['calibration'].update(camera_id='other'))

    def test_outside(self):
        self.run_case('out_of_range', mutate_config=lambda c: c.update(pick_bounds_mm=[0,10,0,10]))

    def test_no_path(self):
        self.run_case('no_feasible_path', mutate_config=lambda c: c.update(path_available=False))

    def test_unknown_execution_stops_and_no_retry(self):
        self.request['simulated_outcomes'] = {'transfer':'unknown'}
        sim = SortingSimulator(self.config)
        result = sim.process(self.request, 1010)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['reason'], 'execution_unknown')
        self.assertEqual([a['action'] for a in result['actions']], ['pick','lift','transfer'])
        self.assertEqual(sim.process(self.request,1010)['reason'], 'duplicate_request')

    def test_each_stage_failure_stops(self):
        for index, step in enumerate(['pick','lift','transfer','release','retreat','verify_source','verify_bin']):
            self.request['simulated_outcomes'] = {step:'failed'}
            result = SortingSimulator(self.config).process(self.request,1010)
            self.assertEqual(len(result['actions']), index+1)
            self.assertEqual(result['reason'], 'step_failed:'+step)

    def test_retired_grip_results_are_not_silently_accepted(self):
        for outcome in ('ok', 'failed', 'unknown'):
            with self.subTest(outcome=outcome):
                self.request['simulated_outcomes'] = {'check_grip': outcome}
                result = SortingSimulator(self.config).process(self.request, 1010)
                self.assertEqual(result['status'], 'rejected')
                self.assertEqual(result['reason'], 'unsupported_automatic_grip_check')
                self.assertEqual(result['actions'], [])

    def test_duplicate_and_old_frame(self):
        sim = SortingSimulator(self.config)
        sim.process(self.request,1010)
        self.assertEqual(sim.process(self.request,1010)['reason'], 'duplicate_request')
        self.request['request_id'] = 'new-request'
        self.request['task_id'] = 'new-task'
        self.assertEqual(sim.process(self.request,1010)['reason'], 'old_frame')

    def test_task_cannot_retry_with_new_request(self):
        sim = SortingSimulator(self.config)
        sim.process(self.request,1010)
        self.request['request_id'] = 'new-request'
        self.assertEqual(sim.process(self.request,1010)['reason'],'duplicate_task')

    def test_observation_failure(self):
        self.run_case('observation_unavailable', lambda r:r.update(observation_error='timeout'))

    def test_malformed_instance_stops(self):
        self.run_case('malformed_input', lambda r:r.update(instances=[None]))

    def test_input_log_is_snapshot(self):
        sim = SortingSimulator(self.config)
        result = sim.process(self.request,1010)
        self.request['instances'].clear()
        self.assertEqual(len(result['input']['instances']),1)
        result['actions'].clear()
        self.assertEqual(len(sim.records[0]['actions']),7)

    def test_non_object_request(self):
        self.assertEqual(SortingSimulator(self.config).process(None,1010)['reason'],'invalid_request_id')

    def test_busy(self):
        sim = SortingSimulator(self.config)
        sim.busy = True
        self.assertEqual(sim.process(self.request,1010)['reason'],'busy')
        self.assertTrue(sim.busy)
        self.assertEqual(sim.seen,set())

    def test_malformed_cases(self):
        cases = [('image_size_mismatch',lambda r:r.update(image_size=[640,480])),
                 ('model_version_mismatch',lambda r:r.update(model_version='changed')),
                 ('invalid_mask',lambda r:r['instances'][0].update(mask_rle=[48,-1])),
                 ('bbox_mask_mismatch',lambda r:r['instances'][0].update(bbox_xyxy=[0,0,1,1])),
                 ('unknown_class',lambda r:r['instances'][0].update(label='other')),
                 ('low_confidence',lambda r:r['instances'][0].update(confidence=float('nan')))]
        original = copy.deepcopy(self.request)
        for reason, mutate in cases:
            self.request = copy.deepcopy(original)
            self.run_case(reason,mutate)


if __name__ == '__main__':
    unittest.main()

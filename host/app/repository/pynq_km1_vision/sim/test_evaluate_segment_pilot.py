import unittest

from src.evaluate_segment_pilot import summarize_frame


class PilotMetricTests(unittest.TestCase):
    def test_correct_lower_confidence_target_does_not_count_as_top_success(self):
        detections = [{'class_id': 2, 'confidence': .9, 'mask_iou': .8},
                      {'class_id': 0, 'confidence': .7, 'mask_iou': .7}]
        result = summarize_frame(0, detections)
        self.assertEqual(result['top_class_and_mask_iou_ge_0_5'], 0)
        self.assertEqual(result['correct_class_and_mask_iou_ge_0_5'], 1)
        self.assertEqual(result['additional_or_unmatched_detections'], 1)

    def test_duplicate_predictions_only_match_one_physical_target(self):
        detections = [{'class_id': 1, 'confidence': .9, 'mask_iou': .5},
                      {'class_id': 1, 'confidence': .8, 'mask_iou': .7}]
        result = summarize_frame(1, detections)
        self.assertEqual(result['class_1_success'], 1)
        self.assertEqual(result['class_1_top_success'], 1)
        self.assertEqual(result['additional_or_unmatched_detections'], 1)

    def test_class_alone_is_not_enough_and_missing_detection_fails(self):
        for detections in ([], [{'class_id': 1, 'confidence': .9, 'mask_iou': .49}]):
            result = summarize_frame(1, detections)
            self.assertEqual(result['class_1_success'], 0)
            self.assertEqual(result['top_class_and_mask_iou_ge_0_5'], 0)

    def test_background_detections_are_false_positives(self):
        result = summarize_frame(None, [{'class_id': 0, 'confidence': .9, 'mask_iou': 0}])
        self.assertEqual(result['empty_frame_false_detections'], 1)
        self.assertNotIn('class_0_success', result)


if __name__ == '__main__':
    unittest.main()

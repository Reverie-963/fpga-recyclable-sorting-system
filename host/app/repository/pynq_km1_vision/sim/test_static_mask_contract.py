"""CPU/recorded HLS interface integration without PYNQ or hardware."""
import copy
import json
from pathlib import Path
import unittest
from src.static_mask_contract import decode_mask, geometry, visual, compare_snapshot
from src.static_sorting import SortingSimulator


class MaskContractTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads((Path(__file__).resolve().parents[1] / 'data/static_sorting/mask_evidence.json').read_text())
        self.r = self.doc['requests'][0]['request']
        self.c = self.doc['config']

    def run_request(self):
        return SortingSimulator(self.c).process(self.r,1010)

    def test_reference_geometry_non_square(self):
        stats = geometry(decode_mask([18,2,6,2,20],[8,6]),[8,6],7)
        self.assertEqual([stats[k] for k in ('m00','m10','m01','m20','m02','m11')],[4,10,10,26,26,25])
        self.assertEqual([stats[k] for k in ('min_row','max_row','min_col','max_col')],[2,3,2,3])

    def test_visual_reference(self):
        stats = visual([1,1,0,0,0,0],[0,1,1,0,0,0],[3,2],1)
        self.assertEqual([stats[k] for k in ('inter','erased','beyond','union')],[1,1,1,3])
        self.assertEqual(stats['max_col'],2)

    def test_no_change_bbox(self):
        stats = visual([0,1],[0,1],[2,1],0)
        self.assertEqual([stats[k] for k in ('min_row','max_row','min_col','max_col')],[0,0,0,0])

    def test_full_evidence_flow(self):
        result = self.run_request()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['verification_stats']['source']['erased'],4)
        self.assertEqual(result['verification_stats']['bin']['beyond'],2)

    def test_source_unchanged_stops_before_bin_check(self):
        self.r['verification_masks']['source']['after_rle'] = self.r['instances'][0]['mask_rle']
        result=self.run_request()
        self.assertEqual(result['reason'],'step_failed:verify_source')
        self.assertEqual(len(result['actions']),6)

    def test_bin_no_gain_stops(self):
        self.r['verification_masks']['bin']['after_rle']=[48]
        result=self.run_request()
        self.assertEqual(result['reason'],'step_failed:verify_bin')
        self.assertEqual(result['status'],'needs_review')

    def test_wrong_evidence_identity(self):
        for key in ('task_id','request_id','camera_id','model_version'):
            with self.subTest(key=key):
                old=self.r['verification_masks'][key]
                self.r['verification_masks'][key]='wrong'
                result=self.run_request()
                self.assertEqual(result['actions'],[])
                self.assertEqual(result['reason'],'evidence_identity_mismatch:'+key)
                self.r['verification_masks'][key]=old

    def test_wrong_bin(self):
        self.r['verification_masks']['bin_id']='plastic_bottle-bin'
        self.assertEqual(self.run_request()['reason'],'evidence_bin_mismatch')

    def test_old_evidence(self):
        self.r['verification_masks']['after_frame_id']=1
        self.assertEqual(self.run_request()['reason'],'evidence_frame_mismatch')

    def test_recorded_geometry(self):
        self.r['geometry_job_id']=7
        self.r['pl_geometry']=geometry(decode_mask(self.r['instances'][0]['mask_rle'],[8,6]),[8,6],7)
        self.assertEqual(self.run_request()['status'],'completed')
        self.r['pl_geometry']['max_col']=7
        self.assertEqual(self.run_request()['reason'],'pl_mismatch:max_col')

    def test_status_and_wrong_job_block(self):
        expected=geometry([1],[1,1],5)
        for key,value in [('status',0),('status',3),('status',5),('job_id',6),('m00',True)]:
            snapshot=dict(expected,**{key:value})
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                compare_snapshot(snapshot,expected)

    def test_recorded_visual(self):
        pair=self.r['verification_masks']['bin']
        pair['pl_stats']=visual(decode_mask(pair['before_rle'],[8,6]),decode_mask(pair['after_rle'],[8,6]),[8,6],12)
        self.assertEqual(self.run_request()['status'],'completed')
        pair['pl_stats']['beyond']=9
        self.assertEqual(self.run_request()['reason'],'pl_mismatch:beyond')

    def test_mask_shape_limits(self):
        for size in ([0,1],[641,1],[8,True],[1.0,1]):
            with self.assertRaises(ValueError): decode_mask([0,1],size)

    def test_source_does_not_match_request(self):
        self.r['verification_masks']['source']['before_rle']=[48]
        self.assertEqual(self.run_request()['reason'],'source_mask_mismatch')

    def test_reference_rejects_truncated_or_nonbinary_pixels(self):
        for invalid in ([], [0], [0, 1, 0], [0, 2], [False, 1]):
            with self.subTest(mask=invalid):
                with self.assertRaises(ValueError): geometry(invalid,[2,1],1)
                with self.assertRaises(ValueError): visual([0,1],invalid,[2,1],1)

    def test_boolean_before_frame_is_not_an_id(self):
        self.r['verification_masks']['before_frame_id'] = True
        self.assertEqual(self.run_request()['reason'],'evidence_frame_mismatch')

    def test_required_evidence_missing(self):
        del self.r['verification_masks']
        self.assertEqual(self.run_request()['reason'],'missing_verification_masks')

    def test_invalid_policy_and_evidence(self):
        self.c['verification_policy']={'min_bin_gain_px':0}
        self.assertEqual(self.run_request()['reason'],'invalid_verification_policy')


if __name__=='__main__':
    unittest.main()

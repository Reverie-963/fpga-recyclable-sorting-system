"""Offline import compatibility with PR33 return dictionaries and owned arrays."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from src.static_driver_replay import replay_bundle, export_result


class DriverReplayTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'data/static_sorting'
        self.bundle=json.loads((base/'driver_capture_fixture.json').read_text())
        self.reference=json.loads((base/'pr33_asset_reference.json').read_text())
        self.entry=self.bundle['requests'][0]

    def run_bundle(self):
        return replay_bundle(self.bundle,self.reference)

    def test_compatible_driver_returns(self):
        out=self.run_bundle()
        self.assertEqual(out['completed'],1)
        self.assertTrue(out['simulation'])
        self.assertEqual(out['provenance'],'synthetic_fixture')
        self.assertEqual(self.entry['captures']['geometry']['stats']['ctrl'],4)

    def test_asset_pair_mismatch(self):
        for name in ('vision.bit','vision.hwh'):
            with self.subTest(name=name):
                saved=self.bundle['assets'][name]
                self.bundle['assets'][name]='0'*64
                with self.assertRaisesRegex(ValueError,'asset_mismatch'):self.run_bundle()
                self.bundle['assets'][name]=saved

    def test_no_summary_as_capture(self):
        with self.assertRaisesRegex(ValueError,'unsupported_capture_schema'):
            replay_bundle({'passed':True,'records':[]},self.reference)

    def test_capture_role_and_identity(self):
        cap=self.entry['captures']['source']
        cap['role']='bin'
        with self.assertRaisesRegex(ValueError,'capture_role'):self.run_bundle()
        cap['role']='source';cap['task_id']='wrong'
        with self.assertRaisesRegex(ValueError,'capture_identity'):self.run_bundle()

    def test_error_status_and_stats(self):
        stats=self.entry['captures']['geometry']['stats']
        stats['status']=3
        with self.assertRaisesRegex(ValueError,'pl_status'):self.run_bundle()
        stats['status']=1;stats['m10']+=1
        with self.assertRaisesRegex(ValueError,'pl_mismatch:m10'):self.run_bundle()

    def test_output_checked_not_only_statistics(self):
        self.entry['captures']['geometry']['output_rle']=[48]
        with self.assertRaisesRegex(ValueError,'capture_pixel_mismatch:geometry'):self.run_bundle()

    def test_visual_output_checked(self):
        self.entry['captures']['source']['output_rle']=[48]
        with self.assertRaisesRegex(ValueError,'capture_pixel_mismatch:source'):self.run_bundle()

    def test_mode_and_morphology_rejected(self):
        geom=self.entry['captures']['geometry'];geom['erode']=1
        with self.assertRaisesRegex(ValueError,'unsupported_morphology'):self.run_bundle()
        geom['erode']=0;self.entry['captures']['bin']['mode']=0
        with self.assertRaisesRegex(ValueError,'unsupported_visual_mode'):self.run_bundle()

    def test_incomplete_and_wrong_size(self):
        cap=self.entry['captures'].pop('geometry')
        with self.assertRaisesRegex(ValueError,'incomplete_capture'):self.run_bundle()
        self.entry['captures']['geometry']=cap;cap['image_size']=[6,8]
        with self.assertRaisesRegex(ValueError,'capture_size'):self.run_bundle()

    def test_duplicate_job(self):
        self.entry['captures']['geometry']['job_id']=11
        self.entry['captures']['geometry']['stats']['job_id']=11
        with self.assertRaisesRegex(ValueError,'invalid_capture_job'):self.run_bundle()

    def test_export_roundtrip_owned_array(self):
        stats={'status':1,'job_id':5}
        out=export_result(stats,[0,1,1,0],[2,2],5,mode=1)
        self.assertEqual(out['output_rle'],[1,2,1])
        stats['status']=0
        self.assertEqual(out['stats']['status'],1)
        with self.assertRaises(ValueError):export_result(stats,[0,1],[2,2],5)

    def test_cli_report_and_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cap=root/'cap.json';ref=root/'ref.json';out=root/'out.json'
            cap.write_text(json.dumps(self.bundle));ref.write_text(json.dumps(self.reference))
            cmd=[sys.executable,'-m','src.static_driver_replay','--capture',str(cap),'--reference-report',str(ref),'--output',str(out)]
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(out.read_text());self.assertEqual(len(report['capture_sha256']),64)
            cap.write_text('{}');out.unlink()
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()

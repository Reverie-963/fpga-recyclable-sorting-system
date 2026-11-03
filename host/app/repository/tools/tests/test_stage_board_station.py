import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]))
import export_board_control
from stage_board_station import ControlExporter


class NativeExporterTests(unittest.TestCase):
    def test_original_grants_and_numeric_qualification_preserved_for_both_authorization_values(self):
        for authorized in (False,True):
            with self.subTest(authorized=authorized),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base=root/'config';base.mkdir()
                image=base/'image.json';image.write_text('{"original":"evidence"}')
                image_ref={'path':'image.json','sha256':export_board_control.sha(image)}
                qualification=base/'qualification.json'
                qualification.write_text(json.dumps({'evidence':[image_ref],'operator_confirmed':True}))
                config={'profiles':{'metal':{'qualification_record':{'path':qualification.name,
                    'sha256':export_board_control.sha(qualification)},'grasp_z_mm':56}},
                    'known':{'experiment_authorized':authorized,'implementation_sources':[]},
                    'arm':{'port':'COM9','usb_identity':{'vid':0x1a86,'pid':0x7523}},
                    'statistics':{'url':'http://127.0.0.1:8780'}}
                source=base/'control.json';source.write_text(json.dumps(config));before=source.read_bytes()
                with patch.object(export_board_control,'ROOT',root):
                    exporter=ControlExporter(root/'local-only/candidate','/dev/ttyUSB0',
                        Path('/home/xilinx/km1-board-control-data/state'))
                    ref=exporter.config(base,{'path':source.name,'sha256':export_board_control.sha(source)})
                    value=json.loads((exporter.output/'objects'/ref['path']).read_text())
                self.assertIs(value['known']['experiment_authorized'],authorized)
                self.assertEqual(value['arm']['usb_identity'],config['arm']['usb_identity'])
                self.assertEqual(value['profiles']['metal']['grasp_z_mm'],56)
                self.assertEqual(value['arm']['port'],'COM9')
                self.assertEqual(value['board_transport'],dict(qualified_port='COM9',arm_device='/dev/ttyUSB0',
                    usb_identity=config['arm']['usb_identity']))
                self.assertEqual(source.read_bytes(),before)
                self.assertEqual(json.loads(qualification.read_text())['operator_confirmed'],True)
                self.assertTrue(value['journal_dir'].endswith('/state/ledger'))


if __name__=='__main__':unittest.main()


import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from src.inference import Predictor,crop_tensor
HAS_ORT=bool(importlib.util.find_spec('onnxruntime') and importlib.util.find_spec('onnx'))
HAS_TRAIN=bool(HAS_ORT and importlib.util.find_spec('torch'))

class ModelTests(unittest.TestCase):
    def test_preprocessing_rgb_nchw(self):
        crop=np.full((10,20,3),[10,20,30],np.uint8)
        out=crop_tensor(crop,64);self.assertEqual(out.shape,(1,3,64,64))
        np.testing.assert_allclose(out[0,:,0,0],[30/255,20/255,10/255],rtol=1e-6)
    @unittest.skipUnless(HAS_ORT,'Optional ONNX packages absent')
    def test_actual_onnx_cpu_and_unavailable_provider(self):
        import onnx
        from onnx import helper,TensorProto
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.onnx'
            node=helper.make_node('ReduceMean',['image'],['logits'],axes=[2,3],keepdims=0)
            graph=helper.make_graph([node],'test',[helper.make_tensor_value_info('image',TensorProto.FLOAT,[1,3,64,64])],[helper.make_tensor_value_info('logits',TensorProto.FLOAT,[1,3])])
            model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=9;onnx.save(model,str(p))
            meta=p.with_suffix('.meta.json');meta.write_text(json.dumps({'labels':['red','green','blue'],'image_size':64,'preprocess':'rgb_nchw_0_1'}))
            cfg={'kind':'onnx','model':str(p),'metadata':str(meta),'providers':['CPUExecutionProvider']}
            predictor=Predictor(cfg);result=predictor.predict(np.full((20,20,3),[0,0,255],np.uint8))
            self.assertEqual(result['label'],'red');self.assertEqual(result['score_kind'],'probability_margin');predictor.close()
            config_path=Path(folder)/'config.json';config_path.write_text(json.dumps(cfg))
            test_data=Path(folder)/'test'
            for label,color in [('red',[0,0,255]),('green',[0,255,0]),('blue',[255,0,0])]:
                dest=test_data/label;dest.mkdir(parents=True);cv2.imwrite(str(dest/'one.png'),np.full((20,20,3),color,np.uint8))
            report_path=Path(folder)/'evaluation.json'
            checked=subprocess.run([sys.executable,'-m','src.evaluate','--config',str(config_path),'--compare',str(config_path),'--data',str(test_data),'--output',str(report_path)],capture_output=True,text=True,timeout=30)
            self.assertEqual(checked.returncode,0,checked.stderr)
            report=json.loads(report_path.read_text());self.assertEqual(report['accuracy'],1.);self.assertEqual(report['label_agreement'],1.)
            cfg['providers']=['DefinitelyUnavailableProvider']
            with self.assertRaises(RuntimeError):Predictor(cfg)
    @unittest.skipUnless(HAS_TRAIN,'Optional CNN training stack absent')
    def test_cnn_training_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for split in ['train','val']:
                for label,color in [('blue',[220,20,20]),('green',[20,220,20])]:
                    p=root/split/label;p.mkdir(parents=True)
                    for i in range(3):cv2.imwrite(str(p/(str(i)+'.png')),np.full((20+i,25,3),color,np.uint8))
            output=root/'classifier.onnx';env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            result=subprocess.run([sys.executable,'-m','src.train_cnn','--data',str(root),'--output',str(output),'--epochs','1','--batch','4'],capture_output=True,text=True,timeout=120,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue(output.exists());report=json.loads(output.with_suffix('.metrics.json').read_text())
            self.assertLess(report['export_cpu_max_abs_error'],1e-4)

if __name__=='__main__':unittest.main(verbosity=2)

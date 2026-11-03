import json,unittest,tempfile,subprocess,sys
from pathlib import Path
import numpy as np,cv2
from src.arm import ik,packet,plan_pick,execute
from src.vision import mask_cpu,candidates,StableTarget,feature,Classifier,map_point
from src.calibrate import fit

class SystemTests(unittest.TestCase):
    def test_mask_boundary(self):
        img=np.array([[[240,240,240],[140,240,240],[139,240,240],[0,0,0]]],np.uint8)
        np.testing.assert_array_equal(mask_cpu(img,[240]*3,100),[[0,0,255,255]])
    def test_blob_center(self):
        img=np.full((120,160,3),240,np.uint8);img[40:70,50:90]=0
        objs=candidates(mask_cpu(img,[240]*3,100),50,2000)
        self.assertEqual(len(objs),1);np.testing.assert_allclose(objs[0]['center'],[69.5,54.5])
    def test_stability_does_not_follow_drifting_object(self):
        s=StableTarget(3,4)
        self.assertFalse(s.update({'label':'a','center':[10,10]}))
        self.assertFalse(s.update({'label':'a','center':[13,10]}))
        self.assertFalse(s.update({'label':'a','center':[16,10]}))
        s.update(None);self.assertFalse(s.update({'label':'a','center':[10,10]}))
        self.assertFalse(s.update({'label':'a','center':[10,10]}));self.assertTrue(s.update({'label':'a','center':[10,10]}))
    def test_protocol(self):
        self.assertEqual(packet([(5,1000)],1500),b'{#005P1000T1500!}')
        for bad in [0,299,5001]:
            with self.assertRaises(ValueError):packet([(5,1000)],bad)
        with self.assertRaises(ValueError):packet([(6,1000)],1000)
    def test_unreachable(self):
        for p in [(0,0,100),(0,-1,30),(0,900,30),(float('nan'),100,30)]:
            with self.assertRaises(ValueError):ik(*p)
    def test_side_positions_and_back_boundary(self):
        calibration={'right_90_pwm':800,'center_pwm':1500,'left_90_pwm':2100}
        right = ik(145, 0, 40, -75, calibration)
        left = ik(-145, 0, 40, -75, calibration)
        self.assertEqual(right[0], 800)
        self.assertEqual(left[0], 2100)
        self.assertEqual(right[1:], left[1:])
        for x in (-145,145):
            with self.assertRaisesRegex(ValueError,'requires base yaw calibration'):
                ik(x,0,40,-75)
        for p in [(145, -1, 40), (-145, -1, 40), (0, 0, 40)]:
            with self.assertRaises(ValueError):
                ik(*p, -75)
    def test_side_bin_can_be_planned_without_execution(self):
        from src.config import validate
        cfg=json.loads(Path('config/baseline_local.json').read_text())
        cfg['arm']['pick_bounds_mm']=[-150,150,0,210]
        cfg['arm']['base_yaw_calibration']={'right_90_pwm':800,'center_pwm':1500,'left_90_pwm':2100}
        validate(cfg)
        for x,expected in [(145,800),(-145,2100)]:
            cfg['arm']['bins']['block']=[x,0,35]
            plan=plan_pick(0,170,'block',cfg)
            above=next(s for s in plan if s[0]=='above_bin')
            self.assertIn(('#000P%04d' % expected).encode(),above[1])
            self.assertEqual(len(execute(plan,cfg)),12)
            side_pick=plan_pick(x,0,'block',cfg)
            self.assertIn(('#000P%04d' % expected).encode(),
                          next(s for s in side_pick if s[0]=='above_target')[1])
            from src.arm_plan import inspect_plan
            cfg['arm'].update(transport='openmv_v3', commissioned=True,
                              joint_limits_pwm=[[900,2000]]*5+[[1000,1700]])
            report=inspect_plan(plan,cfg)
            self.assertTrue(any(s.startswith('above_bin:') for s in report['execution_blockers']))
            cfg['arm']['commissioned']=False
        with self.assertRaises(ValueError):
            plan_pick(0,0,'block',cfg)
        cfg['arm'].pop('base_yaw_calibration')
        cfg['arm']['bins']['block']=[145,0,35]
        with self.assertRaisesRegex(ValueError,'requires base yaw calibration'):
            plan_pick(0,170,'block',cfg)
    def test_optional_base_yaw_calibration_in_plans(self):
        from src.config import validate
        cfg=json.loads(Path('config/baseline_local.json').read_text())
        calibration={'right_90_pwm':800,'center_pwm':1500,'left_90_pwm':2100}
        cfg['arm']['base_yaw_calibration']=calibration
        validate(cfg)
        for x, expected in [(145,800),(-145,2100)]:
            self.assertEqual(ik(x,0,40,-75,calibration)[0],expected)
            cfg['arm']['bins']['block']=[x,0,35]
            plan=plan_pick(0,170,'block',cfg)
            self.assertIn(('#000P%04d' % expected).encode(),
                          next(s for s in plan if s[0]=='above_bin')[1])
        self.assertEqual(ik(0,150,40,-75,calibration)[0],1500)
        self.assertEqual(ik(145,145,40,-75,calibration)[0],1150)
        self.assertEqual(ik(-145,145,40,-75,calibration)[0],1800)
        for bad in [{}, {'right_90_pwm':1500,'center_pwm':1500,'left_90_pwm':2100},
                    {'right_90_pwm':True,'center_pwm':1500,'left_90_pwm':2100}]:
            cfg['arm']['base_yaw_calibration']=bad
            with self.assertRaises(ValueError):
                validate(cfg)
            with self.assertRaises(ValueError):
                ik(145,0,40,-75,bad)
    def test_preflight_and_dry(self):
        cfg=json.loads(Path('config/baseline_local.json').read_text())
        plan=plan_pick(0,170,'block',cfg)
        self.assertEqual(len(execute(plan,cfg)),12)
        self.assertEqual(plan[0][0],'home')
        with self.assertRaises(ValueError):execute(plan,cfg,True)
        cfg['arm']['bins']['block']=[0,900,35]
        with self.assertRaises(ValueError):plan_pick(0,170,'block',cfg)
    def test_calibration(self):
        px=[[10,10],[100,10],[200,10],[10,100],[100,100],[200,100]]
        xy=[[x/2-50,y/2+100] for x,y in px]
        d={'pixels':px,'world_mm':xy,'check_pixels':[[30,30],[70,70],[150,50]],
           'check_world_mm':[[-35,115],[-15,135],[25,125]],'image_size':[640,480],'plane_note':'synthetic'}
        c=fit(d);np.testing.assert_allclose(map_point([50,50],c),[-25,125],atol=1e-5)
        d['check_world_mm'][0]=[999,999]
        with self.assertRaises(ValueError):fit(d)
    def test_model_training_cli(self):
        rng=np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ['train','val']:
                for label,color in [('blue',[210,30,30]),('green',[30,210,30])]:
                    folder=root/split/label;folder.mkdir(parents=True)
                    for i in range(6):
                        img=np.clip(np.array(color)+rng.normal(0,8,(24,30,3)),0,255).astype(np.uint8)
                        cv2.imwrite(str(folder/('%d.png'%i)),img)
            model=root/'model.npz'
            subprocess.run([sys.executable,'-m','src.train','--data',str(root),'--output',str(model)],check=True,capture_output=True)
            m=Classifier(model)
            self.assertEqual(m.predict(np.full((25,30,3),[210,30,30],np.uint8))[0],'blue')
            report=json.loads(model.with_suffix('.metrics.json').read_text());self.assertEqual(report['accuracy'],1.)


    def test_dry_end_to_end(self):
        from unittest.mock import patch
        import os
        from src import baseline_run as run
        cfg=json.loads(Path('config/baseline_local.json').read_text())
        cfg['camera'].update(width=160,height=120)
        cfg['vision']['max_area']=10000
        img=np.full((120,160,3),240,np.uint8);img[45:75,60:100]=0
        class Camera:
            def __init__(self):self.fid=0
            def read(self,after=0,timeout=3):
                self.fid+=1
                return self.fid,__import__('time').monotonic(),img.copy()
            def close(self):pass
        cwd=os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp);Path('data').mkdir()
                weights=np.zeros((770,2));weights[-1,0]=1
                np.savez('data/classifier.npz',mean=np.zeros(769),scale=np.ones(769),weights=weights,labels=np.array(['block','washer']))
                Path('data/calibration.json').write_text(json.dumps({'homography':[[1,0,-80],[0,1,110],[0,0,1]],'image_size':[160,120],
                    'pixel_hull':[[1,1],[158,1],[158,118],[1,118]]}))
                Path('config').mkdir(exist_ok=True)
                Path('config/baseline_local.json').write_text(json.dumps(cfg))
                with patch('src.baseline_run.LatestCamera',return_value=Camera()),patch('sys.argv',['run','--once','--frames','10']),patch('builtins.print'):
                    run.main()
                result=json.loads(Path('board/last_plan.json').read_text())
                self.assertEqual(result['plan'][0]['step'],'home')
                np.testing.assert_allclose(result['xy_mm'],[-.5,169.5])
                self.assertEqual(len(result['plan']),12)
            finally:os.chdir(cwd)

    def test_config_rejects_invalid_hardware_contract(self):
        from src.config import validate
        cfg=json.loads(Path('config/baseline_local.json').read_text());validate(cfg)
        cfg['vision']['threshold']=-1
        with self.assertRaises(ValueError):validate(cfg)


    def test_summary_latest_session_only(self):
        from src.summarize import summarize
        records=[{'event':'start','session':'old'},{'event':'observation','segment_ms':999},
            {'event':'start','session':'new'},{'event':'observation','segment_ms':2,'prediction':{'rpc_ms':3}},
            {'event':'plan','segment_ms':2}]
        summary=summarize(records);self.assertEqual(summary['observations'],1)
        self.assertEqual(summary['segment_ms']['mean'],2);self.assertEqual(summary['events']['plan'],1)

    def test_fused_mask_not_opened_twice(self):
        from unittest.mock import patch
        from src.control import objects_in_area
        mask=np.zeros((30,30),np.uint8);mask[10:20,10:20]=255
        with patch('src.vision.open_mask',side_effect=AssertionError('second morphology')):
            self.assertEqual(len(objects_in_area(mask,{'min_area':10,'max_area':500,'pl_kernel':'open'},None)),1)

if __name__=='__main__':unittest.main(verbosity=2)



"""Timing checks use mocked connection and sleep, with no device access."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from src.arm import plan_pick,execute
from src.arm_plan import inspect_plan

class ArmTimingTests(unittest.TestCase):
    def config(self,fast=False):
        name='arm_pc_fast.example.json' if fast else 'arm_pc.example.json'
        return json.loads((Path(__file__).resolve().parents[1]/'config'/name).read_text())
    def test_legacy(self):
        cfg=self.config();p=plan_pick(0,160,'cap_deformed',cfg)
        self.assertTrue(all(d==1500 for _,_,d in p))
        self.assertAlmostEqual(inspect_plan(p,cfg)['nominal_wait_s'],21.6)
    def test_fast(self):
        cfg=self.config(True);p=plan_pick(0,160,'cap_deformed',cfg)
        for name,data,d in p:
            self.assertIn(('T%04d!'%d).encode(),data)
            self.assertEqual(d,cfg['arm']['step_move_ms'].get(name,1500))
        self.assertLess(inspect_plan(p,cfg)['nominal_wait_s'],21.6)
        self.assertFalse(cfg['arm']['commissioned'])
    def test_full_motion_wait_and_one_connection(self):
        cfg=self.config(True);cfg['arm']['commissioned']=True # mocked connection only
        p=plan_pick(0,160,'cap_deformed',cfg)
        with patch('src.arm_transport.ArmConnection') as conn,patch('src.arm.time.sleep') as sleep:
            execute(p,cfg,True)
            conn.return_value.open.assert_called_once();conn.return_value.close.assert_called_once()
            self.assertEqual(conn.return_value.send.call_count,len(p))
            self.assertEqual(len(sleep.call_args_list),len(p))
            for (name,_,d),call in zip(p,sleep.call_args_list):
                self.assertAlmostEqual(call.args[0],d/1000+cfg['arm']['step_settle_s'][name])
                self.assertGreaterEqual(call.args[0],d/1000)
    def test_bad_parameters(self):
        for key,value in [('step_move_ms',{'unknown':500}),('step_move_ms',{'lift':True}),('step_move_ms',{'close':200}),('step_settle_s',{'close':-.1}),('step_settle_s',{'lift':float('nan')})]:
            cfg=self.config();cfg['arm'][key]=value
            with self.assertRaises(ValueError):plan_pick(0,160,'cap_deformed',cfg)
    def test_no_retry(self):
        cfg=self.config(True);cfg['arm']['commissioned']=True
        p=plan_pick(0,160,'cap_deformed',cfg)
        with patch('src.arm_transport.ArmConnection') as conn,patch('src.arm.time.sleep') as sleep:
            conn.return_value.send.side_effect=TimeoutError('unknown')
            with self.assertRaises(TimeoutError):execute(p,cfg,True)
            self.assertEqual(conn.return_value.send.call_count,1)
            sleep.assert_not_called();conn.return_value.close.assert_called_once()

    def test_trial_preserves_calibration_and_disables_execution(self):
        import copy
        from src.arm_speed_trial import make_trial
        cfg=self.config(True);cfg['arm']['commissioned']=True
        before=copy.deepcopy(cfg)
        trial,report=make_trial(cfg,'candidate',[0,160],'cap_deformed')
        self.assertEqual(cfg,before)
        self.assertFalse(trial['arm']['commissioned'])
        for key in cfg['arm']:
            if key not in ('commissioned','step_move_ms','step_settle_s'):
                self.assertEqual(trial['arm'][key],cfg['arm'][key])
        self.assertAlmostEqual(report['candidate_nominal_s'],11.0)
        self.assertFalse(report['physical_verified'])
        self.assertTrue(report['preview_only'])

    def test_trial_order_and_timing(self):
        from src.arm_speed_trial import make_trial
        _,brisk=make_trial(self.config(),'brisk',[0,160],'cap_deformed')
        _,candidate=make_trial(self.config(),'candidate',[0,160],'cap_deformed')
        self.assertGreater(brisk['candidate_nominal_s'],candidate['candidate_nominal_s'])
        for a,b in zip(brisk['steps'],candidate['steps']):
            self.assertEqual(a['step'],b['step'])
            self.assertGreaterEqual(a['candidate_ms'],b['candidate_ms'])
        self.assertEqual(candidate['steps'][0]['candidate_ms'],1500)

if __name__=='__main__':unittest.main()

import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from src.motion_sequence import scene_check_mode,send_sequence_step


class SequencePolicyTests(unittest.TestCase):
    def test_legacy_default_and_explicit_sequence_mode(self):
        self.assertEqual(scene_check_mode({}),'per_command')
        self.assertEqual(scene_check_mode(dict(motion_scene_check_mode='before_sequence',operator_statement='one complete route')),'before_sequence')
        for options in (dict(motion_scene_check_mode='all_off'),dict(motion_scene_check_mode='before_sequence'),
                        dict(motion_scene_check_mode='before_sequence',operator_statement=True)):
            with self.assertRaises(ValueError):scene_check_mode(options)

    def make_motion(self):
        motion=SimpleNamespace(stopped=False,writes=[],visual=[])
        def stop():
            if motion.stopped:raise ValueError('operator_stop_requested')
        motion.check_stop=stop
        def send(obj,data,duration,guard):
            obj.check_stop();guard();obj.writes.append((data,duration))
        return motion,send

    def test_23_steps_use_one_visual_guard_and_original_bytes(self):
        m,send=self.make_motion()
        for i in range(23):
            send_sequence_step(m,bytes([i]),400,(lambda:m.visual.append('fresh')) if i==0 else None,send)
        self.assertEqual(m.visual,['fresh']);self.assertEqual(m.sequence_scene_checks,1)
        self.assertEqual(m.writes,[(bytes([i]),400) for i in range(23)])

    def test_first_command_without_fresh_guard_is_rejected(self):
        m,send=self.make_motion()
        with self.assertRaisesRegex(ValueError,'initial_visual_guard'):send_sequence_step(m,b'first',400,None,send)
        self.assertFalse(m.writes)

    def test_stale_initial_guard_prevents_all_writes(self):
        m,send=self.make_motion()
        def stale():raise ValueError('frame_expired')
        with self.assertRaisesRegex(ValueError,'frame_expired'):send_sequence_step(m,b'first',400,stale,send)
        self.assertFalse(m.writes);self.assertFalse(getattr(m,'sequence_visual_checked',False))

    def test_stop_between_steps_prevents_next_write(self):
        m,send=self.make_motion();send_sequence_step(m,b'first',400,lambda:None,send);m.stopped=True
        with self.assertRaisesRegex(ValueError,'operator_stop'):send_sequence_step(m,b'second',400,None,send)
        self.assertEqual(m.writes,[(b'first',400)])

    def test_stop_arriving_during_initial_guard_prevents_first_write(self):
        m,send=self.make_motion()
        with self.assertRaisesRegex(ValueError,'operator_stop'):
            send_sequence_step(m,b'first',400,lambda:setattr(m,'stopped',True),send)
        self.assertFalse(m.writes)


@unittest.skipUnless(find_spec('src.board_control_ownership') is not None and find_spec('src.sorting_io') is not None,
                     'requires the deployed native integration foundation')
class NativeSequenceTests(unittest.TestCase):
    def scenario(self,mode='before_sequence'):
        from src.board_station_host import Authority,guard_connection_type
        from src.sorting_relocated import RelocatedMotion
        self.now=0;self.sources=[];self.writes=[];self.stopped=False
        self.owner=SimpleNamespace(blocked=False)
        self.owner.claim=lambda *a:'live-lease'
        def check(lease):
            if self.owner.blocked:raise ValueError('board_control_blocked')
        self.owner.check=check
        a=Authority(self.owner,lambda:self.sources.append('check'),1000,clock=lambda:self.now,motion_scene_check_mode=mode)
        a.begin('test','unique-request');a.last_capture_ms=0
        outer=self
        class Connection:
            mode='direct'
            def __init__(self):self.port=SimpleNamespace(reset_input_buffer=lambda:None)
            def open(self):pass
            def send(self,data,duration,guard=None):
                if guard:guard()
                outer.writes.append(data)
        connection=guard_connection_type(Connection,a)();connection.open()
        m=object.__new__(RelocatedMotion);m.scene_check_mode=mode;m.arm={};m.connection=connection
        def stop():
            if self.stopped:raise ValueError('operator_stop_requested')
        m.check_stop=stop
        m.camera=SimpleNamespace(raw=lambda *a:(_ for _ in ()).throw(AssertionError('unexpected per-step image')))
        return m,a

    def test_real_native_send_has_no_per_step_image_or_file_rescan(self):
        m,a=self.scenario();m.send(b'first',400,lambda:None);self.now=20
        for i in range(22):m.send(bytes([i]),400)
        self.assertEqual(len(self.writes),23);self.assertEqual(len(self.sources),2)
        self.assertEqual(m.sequence_scene_checks,1);self.assertTrue(a.sequence_started)

    def test_real_native_owner_loss_and_stop_still_prevent_write(self):
        for reason in ('owner','stop'):
            with self.subTest(reason=reason):
                m,a=self.scenario();m.send(b'first',400,lambda:None)
                if reason=='owner':self.owner.blocked=True
                else:self.stopped=True
                with self.assertRaises(ValueError):m.send(b'second',400)
                self.assertEqual(self.writes,[b'first'])

    def test_real_native_first_capture_still_must_be_fresh(self):
        m,a=self.scenario();self.now=2
        with self.assertRaisesRegex(ValueError,'capture_expired'):m.send(b'first',400,lambda:None)
        self.assertFalse(self.writes)


if __name__=='__main__':unittest.main()

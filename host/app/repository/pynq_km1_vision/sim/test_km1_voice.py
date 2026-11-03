"""Hardware voice contract: no PC audio, legacy actions, or invented success."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from src.km1_voice import (COMMANDS, REPLIES, parse_request, request_frame,
                          reply_frame, result_reply, check_requested_material)


class KM1VoiceTests(unittest.TestCase):
    def test_requested_material_cannot_relabel_or_bypass_vision(self):
        self.assertEqual(check_requested_material('PAPER','paper'),'paper')
        self.assertEqual(check_requested_material('AUTO','metal'),'metal')
        for token, label in [('PAPER','metal'),('PLASTIC','unknown'),('AUTO',None),('REPORT','paper')]:
            with self.assertRaises(ValueError):check_requested_material(token,label)

    def test_six_exact_hardware_commands(self):
        self.assertEqual([c[1] for c in COMMANDS], ['纸类分拣', '金属分拣', '塑料分拣',
                         '自动分拣', '汇报分拣结果', '停止分拣'])
        for c in COMMANDS:
            self.assertEqual(parse_request(request_frame(c[2])), c[2])
        for f in (b'@KSV1:PAPER', b'@KSV1:PAPER!junk', b'@KSV1:PAPER!@KSV1:AUTO!',
                  b'#ColorSort!', b'$DGT:1-2,1!', b'@KSV1:UNKNOWN!', b'@KSV1:paper!'):
            self.assertIsNone(parse_request(f))

    def test_reply_vocabulary_is_closed(self):
        for token in REPLIES:
            self.assertEqual(reply_frame(token), ('$KSP1:' + token + '!').encode())
        with self.assertRaises(ValueError):
            reply_frame('DONE!$DGT:1-2,1!')

    def test_preview_simulation_and_missing_report_cannot_claim_success(self):
        for mode in ('preview', 'simulation', None):
            self.assertEqual(result_reply(dict(mode=mode, status='completed_visual_evidence',
                                              choice={'label': 'paper'})), 'NO_RESULT')
        for record in (None, {}, {'mode': 'known_physical', 'status': 'unexpected'}):
            self.assertEqual(result_reply(record), 'NO_RESULT')

    def test_raw_review_never_promoted_by_operator_or_supplement(self):
        r=dict(mode='known_physical', status='needs_review', choice={'label': 'paper'},
               operator_review={'resolved': True}, supplemental={'status': 'passed'})
        self.assertEqual(result_reply(r), 'REVIEW')
        self.assertEqual(result_reply(dict(r, status='motion_in_progress')), 'BUSY')

    def test_completed_visual_result_uses_neutral_completion_clip(self):
        self.assertEqual(REPLIES['PAPER_PASS'],'核验完成')
        for label in ('paper','metal','plastic'):
            self.assertEqual(result_reply(dict(mode='known_physical', status='completed_visual_evidence',
                                              choice={'label': label})), 'PAPER_PASS')
        self.assertEqual(result_reply(dict(mode='known_physical', status='completed_visual_evidence',
                                          choice={'label': 'unknown'})), 'NO_RESULT')

    def test_completion_projection_does_not_override_failure_or_preview(self):
        for status,token in [('needs_review','REVIEW'),('rejected','REJECTED'),('motion_in_progress','BUSY')]:
            self.assertEqual(result_reply(dict(mode='known_physical',status=status,voice_verification_completed=True)),token)
        self.assertEqual(result_reply(dict(mode='preview',status='completed_visual_evidence',voice_verification_completed=True)),'NO_RESULT')

    def test_importable_project_keeps_vendor_pins_and_sends_each_request_once(self):
        root=Path(__file__).resolve().parents[2]
        ns={'b': 'http://www.w3.org/1999/xhtml'}
        p=root/'pynq_km1_vision/firmware/asrpro/sorting_voice.hd'
        tree=ET.fromstring(p.read_text(encoding='utf-8-sig'))
        fields=lambda node:{f.get('name'):f.text or '' for f in node.findall('b:field',ns)}
        blocks=list(tree.iter('{'+ns['b']+'}block'))
        ids=[fields(b) for b in blocks if b.get('type')=='asr_id']
        self.assertEqual([d['ASR'] for d in ids], ['小爱同学']+[c[1] for c in COMMANDS])
        self.assertEqual(len(set(d['id'] for d in ids)), 7)
        sends=[fields(b) for b in blocks if b.get('type')=='asr_event_serial_print']
        self.assertEqual([d['NAME'].encode() for d in sends], [request_frame(c[2]) for c in COMMANDS])
        self.assertTrue(all(d['serial_select']=='Serial' and d['wordwrap']=='0' for d in sends))
        uart=next(fields(b) for b in blocks if b.get('type')=='serial_begin' and fields(b)['serial_select']=='Serial1')
        self.assertEqual((uart['TX'],uart['RX']),('PA_2','PA_3'))
        events=[b for b in blocks if b.get('type')=='myasr_serial_readstring']
        self.assertEqual(len(events),len(REPLIES))
        self.assertTrue(all(fields(b)['serial_select']=='Serial' for b in events))
        timeouts=[b for b in blocks if b.get('type')=='serial_setTimeout']
        self.assertEqual(len(timeouts),1)
        self.assertEqual(fields(timeouts[0])['serial_select'],'Serial')
        self.assertEqual(timeouts[0].find('b:value/b:block/b:field',ns).text,'10')
        voices=[fields(b)['audio'] for b in blocks if b.get('type')=='asr_event_play_audio']
        self.assertEqual(voices,list(REPLIES.values()))
        text=p.read_text(encoding='utf-8-sig')
        self.assertNotIn('#ColorSort!',text)
        self.assertNotIn('$DGT:',text)


if __name__ == '__main__':
    unittest.main()

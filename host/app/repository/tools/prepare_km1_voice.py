"""Build a small ASRPRO project from the vendor's unmodified KM1 reference.

This prepares editable sources only. It neither compiles nor flashes hardware.
"""
import argparse
import copy
import csv
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
from src.km1_voice import COMMANDS, REPLIES, request_frame, reply_frame

NS = 'http://www.w3.org/1999/xhtml'
ET.register_namespace('', NS)
WAKE_WORD = '小爱同学'


def elem(tag, **attrs):
    return ET.Element('{' + NS + '}' + tag, attrs)


def block(kind, fields=None):
    b = elem('block', type=kind)
    for name, text in (fields or {}).items():
        f = elem('field', name=name)
        f.text = str(text)
        b.append(f)
    return b


def attach(parent, tag, name, child):
    node = elem(tag, name=name)
    node.append(child)
    parent.append(node)
    return parent


def chain(nodes):
    for left, right in zip(nodes, nodes[1:]):
        n = elem('next')
        n.append(right)
        left.append(n)
    return nodes[0]


def wake():
    return attach(block('set_state_enter_wakeup'), 'value', 'num', block('math_number', {'NUM': 10}))


def project(reference, channel='Serial'):
    if channel not in ('Serial', 'Serial1'):
        raise ValueError('invalid_asr_channel')
    vendor = ET.fromstring(reference.read_text(encoding='utf-8'))
    root = elem('xml')
    callback = block('procedures_defnoreturn', {'NAME': 'ASR_CODE'})
    branches = [wake()]
    for ident, phrase, token, response in COMMANDS:
        compare = block('logic_compare', {'OP': 'EQ'})
        attach(compare, 'value', 'A', block('asr_snid'))
        attach(compare, 'value', 'B', block('math_number', {'NUM': ident}))
        condition = attach(block('controls_if'), 'value', 'IF0', compare)
        send = block('asr_event_serial_print', {'serial_select': channel, 'mode': 'String',
                    'wordwrap': '0', 'NAME': request_frame(token).decode('ascii')})
        attach(condition, 'statement', 'DO0', send)
        branches.append(condition)
    attach(callback, 'statement', 'STACK', chain(branches))
    root.append(callback)
    setup = block('setup')
    initialization = [block('asr_speak', {'VOICE_SET': '小蝶-清新女声', 'vol': '10', 'speed': '10'}),
                      block('asr_welcom', {'NAME': f'用{WAKE_WORD}唤醒，选择分拣功能'}),
                      block('asr_sleep', {'NAME': ''}),
                      block('asr_id', {'ASR': WAKE_WORD, 'leibie': '唤醒词', 'huifu': '我在', 'id': 0})]
    for ident, phrase, token, response in COMMANDS:
        initialization.append(block('asr_id', {'ASR': phrase, 'leibie': '命令词',
                                               'huifu': response, 'id': ident}))
    attach(setup, 'statement', 'DO', chain(initialization))
    root.append(setup)
    # Retain vendor UART0/UART1 pin selections and 115200, plus system delay.
    for b in vendor:
        if b.get('type') in ('hardware_init', 'haodabit_DELAY1'):
            copied=copy.deepcopy(b)
            if b.get('type')=='hardware_init':
                node=copied.find('{'+NS+'}statement').find('{'+NS+'}block')
                while node.find('{'+NS+'}next') is not None:
                    node=node.find('{'+NS+'}next').find('{'+NS+'}block')
                timeout=attach(block('serial_setTimeout',{'serial_select':channel}),
                               'value','timeout',block('math_number',{'NUM':10}))
                nxt=elem('next');nxt.append(timeout);node.append(nxt)
            root.append(copied)
    for token, text in REPLIES.items():
        event = block('myasr_serial_readstring', {'serial_select': channel})
        attach(event, 'value', 'val', block('text', {'TEXT': reply_frame(token).decode('ascii')}))
        audio = block('asr_event_play_audio', {'audio': text})
        attach(event, 'statement', 'DO', chain([wake(), audio]))
        root.append(event)
    # Position top-level blocks vertically for editable import.
    for i, b in enumerate(root):
        b.set('x', '0' if i < 4 else '700')
        b.set('y', str(i * 220))
    return ET.tostring(root, encoding='unicode') + '\n<!-- edittype="asr_pro" -->\n<!-- officialLib="[]" -->\n<!-- personalLib="[]" -->\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'pynq_km1_vision/firmware/asrpro')
    parser.add_argument('--channel', choices=('Serial', 'Serial1'), default='Serial',
                        help='Serial: module USB UART0; Serial1: legacy controller UART, requires its own bridge')
    args = parser.parse_args()
    reference = ROOT/'reference/km1/5.软件工具/7.语音模块配置软件和安装包/Km1_YUYING.hd'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'sorting_voice.hd').write_text(project(reference,args.channel), encoding='utf-8')
    with (args.output/'voice_entries.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['类型', '编号', '词条或触发帧', '模块回复或播报'])
        writer.writerow(['唤醒', 0, WAKE_WORD, '我在'])
        for ident, phrase, token, response in COMMANDS:
            writer.writerow(['命令', ident, phrase, response])
        for token, text in REPLIES.items():
            writer.writerow(['结果播报', '', reply_frame(token).decode('ascii'), text])
    print('Prepared ASRPRO source and voice entries; not compiled, flashed or commissioned.')


if __name__ == '__main__':
    main()

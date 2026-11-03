"""Check portable repository inputs without hardware or third-party packages."""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'pynq_km1_vision'
errors = []
required = [
    'AGENTS.md', 'README.md', '.github/CODEOWNERS',
    'docs/team/START_HERE.md', 'docs/team/STATUS.md',
    'pynq_km1_vision/build/toolchain_version.txt',
    'pynq_km1_vision/build/vivado.tcl',
    'pynq_km1_vision/build/encoder_sen0230_pmodb.xdc',
    'pynq_km1_vision/src/hls/segment_open.cpp',
    'pynq_km1_vision/src/rtl/conveyor_encoder.v',
    'hardware/board_files/pynq-z2/A.0/board.xml',
    'hardware/board_files/pynq-z2/A.0/preset.xml',
    'hardware/board_files/pynq-z2/A.0/part0_pins.xml',
]
for name in required:
    if not (ROOT / name).is_file():
        errors.append(f'Missing required input: {name}')
for folder in ['src', 'sim', 'tools', 'openmv_gripper']:
    for path in (PROJECT / folder).rglob('*.py'):
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')
for path in [PROJECT / 'config.json', *(PROJECT / 'config').glob('*.json')]:
    try:
        json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, OSError) as exc:
        errors.append(f'{path.relative_to(ROOT)}: {exc}')
for path in (ROOT / 'hardware/board_files').rglob('*.xml'):
    try:
        ET.parse(path)
    except ET.ParseError as exc:
        errors.append(f'{path.relative_to(ROOT)}: {exc}')
# Team navigation should work after a clean clone, without local-only assets.
for path in [ROOT / 'README.md', *(ROOT / 'docs/team').glob('*.md')]:
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        target = target.split('#', 1)[0]
        if not target or '://' in target:
            continue
        if not (path.parent / target).exists():
            errors.append(f'Broken navigation link: {path.relative_to(ROOT)} -> {target}')
try:
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    for name in filter(None, tracked):
        path = ROOT / name
        if path.is_file() and path.stat().st_size >= 50 * 1024 * 1024:
            errors.append(f'Git file is >= 50 MiB; use shared assets: {name}')
        if any(part.startswith('.venv') or part in {'.Xil', '__pycache__', 'local-only'} for part in Path(name).parts):
            errors.append(f'Local-only file tracked: {name}')
except (OSError, subprocess.CalledProcessError):
    errors.append('Git index unavailable; run this check inside the cloned repository.')
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print('PASS: repository inputs, Python syntax, configuration, board XML, navigation and tracked-file policy')
print('Scope: no synthesis, GPU training, board access or mechanical action.')

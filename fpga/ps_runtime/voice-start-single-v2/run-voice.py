import argparse,hashlib,json,pathlib,runpy,sys
parser=argparse.ArgumentParser();parser.add_argument('--voice-module',required=True);known,remaining=parser.parse_known_args()
assert known.voice_module=='src.board_voice_host'
base=pathlib.Path('/home/xilinx/km1-board-control-20261031-native24-model-confidence-v2');extension=pathlib.Path(__file__).resolve().parent
assert pathlib.Path(remaining[remaining.index('--root')+1]).resolve()==base
expected={'board_voice_host.py': '0d18ba4c3a7580944a68a67ae3b041e8376c59f45fe57d633c9444709151a549', 'km1_voice_dashboard.py': '22939af9d68a1e227a09231c2ce5e0067028bbb0c4e9d09af3a2a2b47db8afa3'}
for name,h in expected.items():assert hashlib.sha256((extension/'src'/name).read_bytes()).hexdigest()==h,name
scope=json.loads((base/'scope-frozen.json').read_text())
for name,h in scope.items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h,name
sys.path.insert(0,str(base/'pynq_km1_vision'))
import src
src.__path__.insert(0,str(extension/'src'))
sys.argv=[known.voice_module]+remaining
runpy.run_module(known.voice_module,run_name='__main__')

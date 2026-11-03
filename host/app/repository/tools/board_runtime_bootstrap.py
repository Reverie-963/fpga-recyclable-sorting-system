"""Candidate-local composition entry; preserve the installed projection backend."""
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    root=Path(__file__).resolve().parent
    scope=json.loads((root/'scope-frozen.json').read_text())
    for name,digest in scope.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('bootstrap_source_changed:'+name)
    words=[word for word in sys.argv[1:] if word]
    args={words[i][2:]:words[i+1] for i in range(len(words)-1)
          if words[i].startswith('--') and not words[i+1].startswith('--')}
    if Path(args['root']).resolve()!=root:
        raise ValueError('bootstrap_root_changed')
    os.chdir(root)
    sys.path.insert(0,str(root/'pynq_km1_vision'))
    os.environ.update(YOLOE_RECYCLABLE_POLICY=args['policy'],YOLOE_RECYCLABLE_SETTINGS=args['settings'],
        YOLOE_RECYCLABLE_CALIBRATION=args['calibration'],YOLOE_RECYCLABLE_URL=args['vision-url'])
    extension=json.loads((root/'startup-extension.json').read_text())
    preserved=json.loads((root/'startup-preserved-state.json').read_text())
    from src.startup_latency_runtime import install_runtime
    install_runtime(preserved_state=preserved,**extension)
    from src.operation_progress import install_runtime as install_progress
    install_progress()
    sys.argv=['src.board_station_host']+words
    from src.board_station_host import main as serve
    serve()


if __name__=='__main__':main()

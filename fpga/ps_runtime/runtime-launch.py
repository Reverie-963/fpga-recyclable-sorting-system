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
    preflight='--startup-preflight' in words
    words=[w for w in words if w!='--startup-preflight']
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
    from src.startup_preparation_runtime import install_registered_policy_validation
    install_registered_policy_validation()
    from src.motion_path_runtime import install_runtime as install_motion
    install_motion(**json.loads((root/'motion-extension.json').read_text()))
    from src.startup_preparation_runtime import install_runtime as install_preparation
    proof=install_preparation(**json.loads((root/'startup-optimization.json').read_text()))
    from src.operation_progress import install_runtime as install_progress
    install_progress()
    if preflight:
        from src.execution_models import ExecutionModels
        from src.board_station_host import initial_decision_thresholds
        from src.sorting_prompted_motion import calibrated_settings,read_motion_options
        selected=ExecutionModels(json.loads((root/'execution-models.json').read_text()),Path(args['state-root'])/'execution-model-binding.json')
        thresholds=initial_decision_thresholds(read_motion_options(),selected.context,calibrated_settings)
        assert thresholds['min_confidence']==selected.snapshot()['settings']['confidence']
        proof['selected_profile_bootstrap_check']={'passed':True,'profile_id':selected.snapshot()['id'],'confidence':thresholds['min_confidence'],'physical_motion':False}
        import runpy
        proof['contour_grasp_check']=runpy.run_path(str(root/'contour-preflight.py'))['check'](root)
        proof['rgb_lut_check']=runpy.run_path(str(root/'lut-preflight.py'))['check']()
        print(json.dumps(dict(status='startup_preflight_passed',**proof)),flush=True)
        return
    from src.monitor_prompted_operation import PromptedBackend
    from src.prompted_height import object_height
    original_init=PromptedBackend.__init__
    def restore_height(self,*a,**kw):
        original_init(self,*a,**kw)
        self.object_height_mm=object_height(preserved.get('object_height_mm'))
    PromptedBackend.__init__=restore_height
    sys.argv=['src.board_station_host']+words
    from src.board_station_host import main as serve
    serve()


if __name__=='__main__':main()

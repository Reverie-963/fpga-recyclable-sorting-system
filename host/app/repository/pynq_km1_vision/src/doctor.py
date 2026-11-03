"""Read-only diagnostics; no automatic driver or firmware installation."""
import importlib
import contextlib
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata


def main():
    result = {'os': platform.platform(), 'machine': platform.machine(),
              'python': sys.version, 'tools': {}, 'packages': {}}
    for name in ['vivado', 'vitis_hls', 'vitis-run', 'nvidia-smi']:
        result['tools'][name] = shutil.which(name)
    for name in ['numpy', 'cv2', 'serial', 'pynq', 'onnxruntime', 'torch', 'torchvision', 'ultralytics']:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                module = importlib.import_module(name)
            result['packages'][name] = getattr(module, '__version__', 'installed')
            if name == 'onnxruntime':
                result['ort_available_providers'] = module.get_available_providers()
            if name == 'torch':
                result['cuda_available'] = module.cuda.is_available()
                result['torch_cuda_build'] = module.version.cuda
                result['torch_hip_build'] = module.version.hip
                result['cuda_devices'] = [module.cuda.get_device_name(i)
                                          for i in range(module.cuda.device_count())]
        except Exception as exc:
            result['packages'][name] = type(exc).__name__ + ': ' + str(exc)
    result['ort_distributions'] = {}
    for name in ['onnxruntime', 'onnxruntime-gpu']:
        try:
            result['ort_distributions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    if len(result['ort_distributions']) > 1:
        result['ort_warning'] = 'CPU/GPU wheels conflict: reinstall only one in this environment'
    if result['tools']['nvidia-smi']:
        try:
            check = subprocess.run([result['tools']['nvidia-smi'],
                '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10)
            result['nvidia_smi'] = {'returncode': check.returncode,
                                    'output': check.stdout.strip(), 'error': check.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result['nvidia_smi'] = {'error': str(exc)}
    result['note'] = 'Inventory only; available provider is not a successful GPU model execution.'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

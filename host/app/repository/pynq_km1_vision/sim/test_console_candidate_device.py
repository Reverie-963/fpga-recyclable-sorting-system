"""A requested GPU must fail explicitly if ROCm is absent; CPU is opt-in."""
from types import SimpleNamespace
import unittest
from src.console_candidate_monitor import runtime_execution


def fake_torch(hip='rocm-test', available=True):
    return SimpleNamespace(__version__='test', version=SimpleNamespace(hip=hip),
        cuda=SimpleNamespace(is_available=lambda:available, get_device_name=lambda _: 'Radeon 890M'),
        backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)))


class DeviceTests(unittest.TestCase):
    def test_gpu_reports_actual_rocm_device_and_fp16(self):
        torch=fake_torch()
        actual=runtime_execution(torch,'cuda:0')
        self.assertEqual((actual['device'],actual['precision'],actual['device_name']),('cuda:0','fp16','Radeon 890M'))
        self.assertFalse(torch.backends.cudnn.enabled)

    def test_missing_rocm_or_gpu_does_not_fall_back(self):
        for torch in (fake_torch(hip=None),fake_torch(available=False)):
            with self.assertRaisesRegex(RuntimeError,'no silent CPU fallback'):
                runtime_execution(torch,'cuda:0')

    def test_cpu_is_explicit_and_fp32(self):
        actual=runtime_execution(fake_torch(hip=None,available=False),'cpu')
        self.assertEqual((actual['device'],actual['precision']),('cpu','fp32'))
        with self.assertRaises(ValueError):runtime_execution(fake_torch(),'auto')

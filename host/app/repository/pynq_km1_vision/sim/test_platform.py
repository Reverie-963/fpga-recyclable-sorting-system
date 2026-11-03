"""Platform integration contracts without requiring a physical GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from src.inference import Predictor
from src.benchmark import summary
from src.summarize import summarize


class PlatformTests(unittest.TestCase):
    def test_cuda_dll_loading_precedes_session_and_failed_provider_is_rejected(self):
        events = []
        class Session:
            def __init__(self, *args, **kwargs):
                events.append('session')
                assert events == ['preload', 'session']
            def disable_fallback(self):
                pass
            def get_providers(self):
                return ['CPUExecutionProvider']
        fake = SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider'],
            preload_dlls=lambda: events.append('preload'), SessionOptions=SimpleNamespace,
            InferenceSession=Session)
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'model.onnx'
            meta = Path(folder)/'model.meta.json'
            model.write_bytes(b'mock model')
            meta.write_text(json.dumps({'image_size':64, 'labels':['a','b'],
                                       'preprocess':'rgb_nchw_0_1'}))
            with patch.dict('sys.modules', {'onnxruntime':fake}):
                with self.assertRaisesRegex(RuntimeError, 'failed to initialize'):
                    Predictor({'kind':'onnx','model':str(model),'metadata':str(meta),
                               'providers':['CUDAExecutionProvider']})
        self.assertEqual(events, ['preload','session'])

    def test_required_percentiles(self):
        result = summary([1, 2, 3, 4, 100])
        self.assertEqual(result['p50_ms'], 3)
        self.assertGreater(result['p99_ms'], result['p95_ms'])
        result = summarize([{'event':'observation','segment_ms':v} for v in [1,2,3,4,100]])
        self.assertEqual(result['segment_ms']['p50'], 3)
        self.assertEqual(result['segment_ms']['max'], 100)


if __name__ == '__main__':
    unittest.main()

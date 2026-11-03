"""Local HTTP replay of the annotated exercise; does not access board or actuators."""
import json
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

import cv2
import onnx

from src.detect_remote import DetectionClient
from src.detect_server import make_server
from src.detection import YOLODetector


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cfg = json.loads(Path('config/host_cap_deformation_cuda.json').read_text(encoding='utf-8-sig'))
    detector = YOLODetector(cfg)
    assert detector.labels == ['cap_ok', 'cap_deformed']
    assert detector.providers == ['cuda_pytorch']
    onnx.checker.check_model('data/models/cap_deformation/best.onnx')
    token = secrets.token_urlsafe(32)
    env_key = 'KM1_LOCAL_DEMO_CHECK_TOKEN'
    previous = os.environ.get(env_key)
    os.environ[env_key] = token
    server = make_server(('127.0.0.1', 0), detector, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}'
        request = Request(url + '/health', headers={'Authorization': 'Bearer ' + token})
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            health = json.load(response)
        assert health['protocol'] == 2 and health['task'] == 'detect'
        client = DetectionClient({'url': url, 'token_env': env_key, 'timeout_s': 5, 'max_result_age_s': 5})
        manifest = json.loads(Path('report/cap_deformation_demo_dataset_20261007.json').read_text())
        results = []
        for fid, item in enumerate(manifest['images']):
            frame = cv2.imread(item['source'])
            assert frame is not None
            result = client.predict(frame, fid, time.monotonic())
            expected = manifest['labels'][item['class_id']]
            matched = any(d['label'] == expected and iou(d['xyxy'], item['bbox_xyxy']) >= 0.5 for d in result['detections'])
            results.append({'source': item['source'], 'split': item['split'], 'expected': expected,
                            'expected_box_found_iou50': matched, 'response': result})
        output = {'record_date': '2026-10-07', 'scope': 'PC loopback HTTP replay of same-specimen exercise only',
                  'independent_accuracy': False, 'board_network_verified': False, 'fresh_camera_verified': False,
                  'onnx_structural_check': 'passed', 'health': health, 'images': results,
                  'matched': sum(r['expected_box_found_iou50'] for r in results), 'count': len(results)}
        Path('report/cap_deformation_local_inference_20261007.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in output.items() if k != 'images'}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


if __name__ == '__main__':
    main()

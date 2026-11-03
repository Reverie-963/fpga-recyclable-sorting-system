"""Prepared YOLOv8s monitor entry; only runs when explicitly launched."""
import argparse
from pathlib import Path
import numpy as np
from .yolov8s_inference import Yolov8sDetector, ROOT
from .yoloe_preview_server import make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8778)
    parser.add_argument('--camera-url', default='http://127.0.0.1:8780/frame.jpg')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'local-only/yolov8s-deployment/preview')
    args = parser.parse_args()
    detector = Yolov8sDetector(half=not args.fp32)
    detector.predict(np.zeros((480, 640, 3), np.uint8))
    server = make_server(detector, args.port, args.output_dir,
        ROOT / 'local-only/yoloe-deployment/examples', args.camera_url)
    server.live.start()
    print('YOLOv8s ROCm monitor ready: http://127.0.0.1:' + str(server.server_port), flush=True)
    try:
        server.serve_forever()
    finally:
        server.live.close()
        server.server_close()


if __name__ == '__main__':
    main()

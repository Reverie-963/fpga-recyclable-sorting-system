"""Frozen local YOLOv8s-seg adapter; shares the trained-material protocol."""
from .yolo11s_inference import Yolo11sDetector, ROOT

ASSETS = ROOT / 'local-only/yolov8s-deployment/assets'


class Yolov8sDetector(Yolo11sDetector):
    MODEL_FAMILY = 'yolov8s'
    MODEL_NAME = 'YOLOv8s-seg'
    LOCK_SCHEMA = 'yolov8s-monitor-deployment-v1'

    def __init__(self, assets=ASSETS, *, device='cuda:0', half=True):
        super().__init__(assets, device=device, half=half)

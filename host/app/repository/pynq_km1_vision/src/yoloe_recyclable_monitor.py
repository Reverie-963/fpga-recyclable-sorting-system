"""Official YOLOE + TACO original labels, grouped for sorting preview."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import threading
from .prompted_recyclables import read_policy, settings_for, adapt
from .empty_background_records import EmptyBackgroundRecords
from .yoloe_official_monitor import OfficialDetector, Playground, make_server, ROOT, ORIGINAL


class RecyclableDetector:
    def __init__(self, engine, policy):
        self.engine, self.policy = engine, copy.deepcopy(policy)
        self.execution, self.hashes = engine.execution, engine.hashes
        self.embeddings = engine.embeddings
        self.vocabulary = [r['official_label'] for r in policy['classes']]

    def prepare(self, payload):
        settings = settings_for(self.policy, payload)
        self.engine.prepare(settings)
        return settings

    def predict(self, frame, settings):
        return adapt(self.policy, settings, self.engine.predict(frame, settings))


class RecyclablePlayground(Playground):
    def __init__(self, detector, camera_url, state_path, port):
        # Base initialization uses prepare(mode=free); initialize with a small
        # facade rather than making the official experiment's default change.
        class Initial:
            def prepare(self, _):
                return detector.prepare({})
        super().__init__(Initial(), camera_url)
        self.detector = detector
        self.enabled = False
        self.state_path, self.port = Path(state_path), port
        self.plan_lock = threading.Lock()
        self.settings_lock = threading.Lock()
        self.background_records = EmptyBackgroundRecords(self.state_path.parent/'empty-backgrounds')
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text(encoding='utf-8'))
            if saved.get('policy_sha256') != self.settings['policy_sha256']:
                raise ValueError('已保存设置的类别策略发生变化，请先核对配置')
            self.configure(dict(saved, enabled=False))

    def configure(self, payload):
        with self.settings_lock:
            return self._configure(payload)

    def _configure(self, payload):
        # Keep unspecified controls when pausing/resuming live recognition.
        with self.lock:
            merged = copy.deepcopy(self.settings)
        for key in ('heights_mm', 'class_thresholds', 'material_thresholds'):
            if key in payload and isinstance(payload[key], dict):
                values = copy.deepcopy(merged[key])
                for name, value in payload[key].items():
                    if key == 'material_thresholds' and isinstance(value, dict):
                        values[name] = dict(values.get(name, {}), **value)
                    else:
                        values[name] = value
                payload = dict(payload, **{key: values})
        merged.update(payload)
        # Public overrides use material_thresholds; all resulting settings are
        # snapshotted, including prompts and heights, before publishing them.
        result = super().configure(merged)
        with self.lock:
            saved = dict(self.settings)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.state_path)
        return result

    def backgrounds(self):
        with self.lock:
            frame = self.frame
        stream = frame['identity'][0] if self.fresh(frame) else None
        return dict(records=self.background_records.list(), current_stream_id=stream,
                    model_trained=False, used_for_automatic_filtering=False)

    def record_background(self, payload):
        with self.lock:
            packet = self.frame if self.fresh(self.frame) else None
            settings_hash = self.settings['settings_sha256']
        return self.background_records.save(packet, confirmed_empty=payload.get('confirmed_empty'),
                                            settings_sha256=settings_hash)

    def plan(self, payload):
        if payload:
            raise ValueError('路线预览使用当前已应用设置和实时相机')
        if not self.plan_lock.acquire(blocking=False):
            raise ValueError('已有路线预览正在处理')
        try:
            with self.lock:
                settings = copy.deepcopy(self.settings)
            interpreter = ROOT / '.venv/Scripts/python.exe'
            run = subprocess.run([str(interpreter), '-m', 'src.sorting_prompted_preview',
                '--url', 'http://127.0.0.1:'+str(self.port)], cwd=ROOT/'pynq_km1_vision',
                input=json.dumps(settings), text=True, encoding='utf-8', capture_output=True,
                timeout=100, env=dict(os.environ, PYTHONUTF8='1'))
            if run.returncode:
                raise ValueError('路线预览未通过：'+run.stderr[-500:])
            result = json.loads(run.stdout)
            with self.lock:
                if settings['settings_sha256'] != self.settings['settings_sha256']:
                    raise ValueError('预览期间设置已改变，请重新预览')
            return result
        finally:
            self.plan_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8811)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--camera-url', default='http://127.0.0.1:8780/frame.jpg')
    args = parser.parse_args()
    os.chdir(ORIGINAL)
    detector = RecyclableDetector(OfficialDetector(args.device), read_policy())
    playground = RecyclablePlayground(detector, args.camera_url,
        ROOT/'local-only/yoloe-recyclables/settings.json', args.port)
    server = make_server(playground, args.port, 'yoloe_recyclables.html')
    playground.start()
    print('TACO prompted preview ready (live paused): http://127.0.0.1:'+str(args.port), flush=True)
    try:
        server.serve_forever()
    finally:
        playground.done.set()
        server.server_close()


if __name__ == '__main__':
    main()

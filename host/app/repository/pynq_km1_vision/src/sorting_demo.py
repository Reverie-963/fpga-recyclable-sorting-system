"""Executable three-material synthetic loop. No device or neural-model access."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .sorting_loop import Controller, Journal, LABELS
from .sorting_io import Statistics


def encode(mask):
    values = mask.astype(np.uint8).reshape(-1)
    edges = np.flatnonzero(values[1:] != values[:-1])+1
    counts = np.diff(np.r_[0, edges, len(values)]).tolist()
    return ([0] if values[0] else []) + counts


def demo_config():
    c = dict(schema_version='sorting-loop-v1', synthetic_fixture=True,
        station_id='synthetic-three-material-station', camera_id='synthetic-camera',
        image_size=[160, 120], model_version='synthetic-masks-no-model', runtime_id='synthetic-runtime',
        calibration_id='synthetic-mm-map', prompt_mode='visual', stable_frames=3,
        max_objects=3, max_age_ms=2000, frame_timeout_s=2, min_frame_gap_ms=100,
        min_separation_px=5, min_interior_px=2, max_cycle_s=120, observe_timeout_s=15,
        min_confidence=.8, min_scene_iou=.9, min_erased_fraction=.9,
        min_persistent_fraction=.8, max_scene_changed_fraction=.02,
        max_other_bin_changed_fraction=.02, rgb_difference_threshold=90,
        min_bin_gain_px=80, retreat_settle_s=.1, require_distinct_jpeg=True,
        quality_gate_passed=False, station_commissioned=False, initial_pose_confirmed=False,
        source_roi=[5, 30, 80, 110], statistics=dict(backend='cpu_reference'),
        layout=dict(static_rois=[[2, 2, 30, 22], [130, 2, 158, 22]], search_margin_px=3,
                    max_layout_shift_px=1, min_patch_std=8, min_patch_corners=4,
                    min_patch_peak_margin=.01, min_patch_correlation=.9, max_patch_mean_difference=12),
        arm=dict(transport='direct', commissioned=False, port='SYNTHETIC_NO_SERIAL',
                 move_ms=1500, settle_s=.1, home=[0, 150, 70], home_alpha_deg=-75,
                 pick_bounds_mm=[-70, 70, 100, 210], joint_limits_pwm=[[500, 2500]]*6),
        profiles={})
    for i, label in enumerate(LABELS):
        c['profiles'][label] = dict(bin_id='bin_'+label, bin_roi=[90, 25+i*30, 155, 50+i*30],
            commissioned=False, swept_path_verified=False, acceptance_record=None,
            object_profile_id='synthetic-'+label, object_height_mm=50,
            image_to_grip_mm=[[.8, 0, -36], [0, .5, 120], [0, 0, 1]],
            image_to_hover_mm=[[.8, 0, -36], [0, .5, 120], [0, 0, 1]],
            verified_polygon_px=[[5, 30], [79, 30], [79, 109], [5, 109]],
            grasp_z_mm=30, hover_z_mm=70, bin_xyz_mm=[-60+i*60, 120, 35],
            alpha_deg=-75, wrist_pwm=1500, open_pwm=1000, close_pwm=1500,
            axis_deg=90, axis_tolerance_deg=10, min_area_px=200, max_area_px=1000,
            min_width_mm=3, max_width_mm=20)
    return c


class DemoClock:
    def __init__(self):
        self.seconds = 10.

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


class DemoSource:
    def __init__(self, c, clock):
        self.c, self.clock = c, clock
        self.frame_id = 0
        self.barrier = -1
        self.stream = 'synthetic-stream'
        self.items = []
        for i, label in enumerate(LABELS):
            mask = np.zeros((120, 160), bool)
            mask[45:85, 14+i*24:24+i*24] = True
            self.items.append(dict(label=label, mask=mask, placed=False))
        self.reference = np.full((120, 160, 3), 100, np.uint8)
        rng = np.random.default_rng(20261029)
        for x0, y0, x1, y1 in c['layout']['static_rois']:
            self.reference[y0:y1, x0:x1] = rng.integers(0, 256, (y1-y0, x1-x0, 3), dtype=np.uint8)

    def after(self, stamp):
        self.barrier = stamp

    def next(self, timeout):
        self.clock.sleep(.2)
        self.frame_id += 1
        image = self.reference.copy()
        objects = []
        for i, item in enumerate(self.items):
            image[item['mask']] = [220, 230-i*20, 240]
            ys, xs = np.nonzero(item['mask'])
            objects.append(dict(label=item['label'], confidence=.97, instance_id=str(i),
                bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                mask_rle=encode(item['mask'])))
        # Unique synthetic camera bytes outside the action and reference ROIs.
        image[119, 0] = [self.frame_id % 255, self.frame_id // 255, 0]
        sha = hashlib.sha256(image.tobytes()).hexdigest()
        obs = {k: self.c[k] for k in ('camera_id', 'model_version', 'runtime_id', 'image_size', 'prompt_mode')}
        obs.update(frame_id=self.frame_id, capture_started_ms=self.clock.monotonic()*1000,
                   source_stream_id=self.stream, image_sha256=sha, instances=objects)
        return dict(image=image, observation=obs, jpeg_sha256=sha, bgr_sha256=sha)


class DemoMotion:
    def __init__(self, source, clock):
        self.source, self.clock = source, clock
        self.opened = False
        self.sent = []
        self.names = []
        self.fail_step = None
        self.wrong_bin = False
        self.item = None

    def check_stop(self):
        pass

    def open(self, plan):
        self.opened = True
        self.names = [n for n, _, _ in plan]
        self.item = min((i for i in self.source.items if not i['placed']), key=lambda x: x['label'])

    def send(self, data, duration, guard=None):
        if guard:
            guard()
        name = self.names.pop(0)
        self.sent.append(name)
        if name == self.fail_step:
            raise IOError('Synthetic partial serial write')
        if name == 'release':
            label = self.item['label']
            if self.wrong_bin:
                label = LABELS[(LABELS.index(label)+1) % 3]
            x, y, _, _ = self.source.c['profiles'][label]['bin_roi']
            self.item['mask'][:] = False
            self.item['mask'][y+5:y+15, x+5:x+45] = True
            self.item['placed'] = True

    def wait(self, seconds):
        self.clock.sleep(seconds)

    def close(self):
        self.opened = False


def build_demo(directory, mode='simulation'):
    c, clock = demo_config(), DemoClock()
    source = DemoSource(c, clock)
    motion = DemoMotion(source, clock)
    journal = Journal(directory)
    controller = Controller(c, source, Statistics(), motion, journal, source.reference,
                            mode=mode, clock=clock.monotonic, sleep=clock.sleep)
    return controller, source, motion, journal


def run_demo(directory):
    controller, source, motion, journal = build_demo(directory)
    outcomes = []
    with journal:
        for _ in range(4):
            result = controller.cycle()
            outcomes.append(result)
            if result['status'] != 'simulated_visual_pass':
                break
        summary = dict(simulation=True, physical_motion=False,
            completed=sum(r['status'] == 'simulated_visual_pass' for r in outcomes),
            labels=[r['choice']['label'] for r in outcomes if 'choice' in r],
            statuses=[r['status'] for r in outcomes], hardware_access=False)
        journal.save('demo-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = run_demo(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['completed'] == 3 and summary['statuses'][-1] == 'empty_scene' else 2


if __name__ == '__main__':
    raise SystemExit(main())

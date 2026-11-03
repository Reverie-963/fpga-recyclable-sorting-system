import hashlib
import json
import pathlib
import sys
import cv2
import numpy as np

OUT = pathlib.Path(__file__).parent
ROOT = pathlib.Path('C:/FPGA')
sys.path.insert(0, str(ROOT / 'pynq_km1_vision'))
from src.inventory_reference import ReferenceBank
from src.recognition_profiles import configuration_hash


def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def stats(values, p99_min=100):
    v = np.asarray(values, dtype=float)
    return dict(n=len(v), p50_ms=float(np.percentile(v, 50)),
                p95_ms=float(np.percentile(v, 95)),
                p99_ms=float(np.percentile(v, 99)) if len(v) >= p99_min else None,
                max_ms=float(v.max()), min_ms=float(v.min()))


live = read(OUT / 'live-samples.json')
rows = live['rows']
ok = [r for r in rows if r.get('http_status') == 200]
valid = [r for r in ok if r.get('backend') == 'pl_board_local' and r.get('available') and not r.get('faulted')]
assert len({r['stream_id'] for r in valid}) == 1
unique = {r['frame_id']: r for r in valid}
urows = sorted(unique.values(), key=lambda r: r['received_monotonic_s'])
span = urows[-1]['received_monotonic_s'] - urows[0]['received_monotonic_s']
fps = (urows[-1]['frame_id'] - urows[0]['frame_id']) / span
comm_span = rows[-1]['received_monotonic_s']-rows[0]['sent_monotonic_s']
summary = dict(
    schema='competition-metrics-audit-v1', date='2026-11-01',
    quantile_method='numpy.percentile linear; P99 omitted for n<100; 100 samples do not establish stable tail latency',
    communication=dict(scope=live['scope'], started_at=live['started_at'], finished_at=live['finished_at'],
                       attempts=len(rows), success=len(ok), failures=len(rows)-len(ok),
                       observed_application_error_percent=100*(len(rows)-len(ok))/len(rows),
                       observed_timeout_percent=100*sum('timed out' in r.get('error','').lower() for r in rows)/len(rows),
                       deadline_s=live['request_timeout_s'], automatic_retry=False,
                       payload_bytes=sum(r['response_bytes'] for r in ok), duration_s=comm_span,
                       observed_throttled_application_payload_mbps=sum(r['response_bytes'] for r in ok)*8/comm_span/1e6,
                       payload_rate_scope='100 vision JSON replies, includes deliberate 250ms pauses; observed workload rate, not link capacity or high-bandwidth test',
                       latency=stats([r['rtt_ms'] for r in ok]), packet_loss_percent=None,
                       wire_only_latency=None, one_way_latency=None, bulk_link_throughput=None,
                       safe_state_unchanged=live['before']==live['after']),
    live_PL=dict(scope='PS+PL input packing/cache synchronization/compute/readback, current background load; not pure kernel latency',
                 latency=stats([r['processing_ms'] for r in urows]), unique_frames=len(urows),
                 observation_span_s=span, observed_camera_PS_PL_processed_fps=fps,
                 observed_camera_PS_PL_pixels_per_s=fps*640*480,
                 frame_drop_rate=None, real_time_deadline_miss_rate=None,
                 assets=live['health_before']['assets']),
    active=live['before'],
    limitations=['No motion or POST requests in this audit',
                 'Current needs_review state retained',
                 'No formal full-task success denominator or physical-response timestamp samples',
                 'No power measurements, clean-environment deployment benchmark, or second-device reuse trial'])

ab = read(ROOT/'local-only/pc-camera-ab-20261101/board-results.json')
samples = [r for r in ab['samples'] if not r['warmup']]
summary['AI_timing'] = {k: stats([r[k] for r in samples]) for k in
                        ['fetch_or_read_ms', 'decode_ms', 'detect_http_ms', 'request_to_result_ms', 'model_inference_ms']}
summary['AI_timing_scope'] = dict(n=10, warmup_excluded=2, input=[640,480], instances_in_all_measured_samples=0,
                                profile=ab['model'], date='2026-11-01',
                                internal_inference_field_is_not_separated_pure_network_time=True,
                                preprocessing_ms=None, pure_model_ms=None, postprocessing_ms=None,
                                total_perception_excludes=ab['exclusions'], sustained_inference_fps=None)

routefile = ROOT/'docs/team/reviews/2026-10-31-fpga-speed-retest.json'
route = read(routefile)
summary['route_compilation'] = dict(
    scope=route['scope'], date=route['date'], runtime=route['runtime'],
    stats={m:stats([r['times_ms'][m] for r in route['rows']]) for m in ['scalar','numpy','pl']},
    speedup_over_optimized_CPU=route['speedup_over_optimized_cpu'],
    speedup_over_scalar_CPU=route['speedup_over_original_scalar'],
    all_command_sequences_equal=route['command_sequences_equal'],
    not_whole_system_speedup=True)

buildfile = pathlib.Path('C:/FPGA-pl-motion/local-only/motion-integration-current/final-evidence/joint-build.json')
build = read(buildfile)
assert set(build['assets'].values()) == set(summary['live_PL']['assets'].values())
summary['FPGA_build'] = dict(clock_mhz=build['clock_mhz'], part=build['part'], toolchain=build['toolchain'],
                            assets=build['assets'], timing=build['timing'], resources=build['resources'],
                            resource_available={'Slice LUTs':53200,'Slice Registers':106400,'Block RAM Tile':140,'DSPs':220},
                            distributed_memory_LUTs=1312, distributed_memory_available_LUTs=17400,
                            slice_occupancy_percent=56.76,
                            drc_errors=0, drc_warnings=57, methodology_warning_LUTAR_1=11,
                            measured_running_clock_mhz=None,
                            provenance='joint-build summary plus jm2 utilization.rpt and timing_after.rpt; asset hashes match current live service')

# Offline replay: use stored CPU FP32 masks, frozen runtime reference bank and
# current profile threshold. No neural inference, training, or hardware calls.
models = read(ROOT/'local-only/model-switch-console/models.json')
entry = next(m for m in models['models'] if m['id']=='yoloe26_reference')
assert configuration_hash(entry) == live['before']['execution_model']['configuration_sha256']
assert sha(entry['weights']) == live['before']['execution_model']['model_sha256']
bank = ReferenceBank(entry['reference_bank'], entry['reference_bank_sha256'])
base = ROOT/'local-only/yoloe-public16-web-20261031/reference-comparison'
features = read(base/'yoloe26-public-retrained-features.json')
stored = {r['image']:r for r in [json.loads(s) for s in
    (ROOT/'local-only/yoloe-public16-web-20261031/local-evaluation/candidate-predictions.jsonl').read_text(encoding='utf-8').splitlines()]}
dataset = ROOT/'local-only/recyclables-final-16-v1'
frozen_inputs = read(base/'inputs.json')
frozen_inputs = {str(pathlib.Path(k)).lower():v for k,v in frozen_inputs.items()}


def verify_frozen(p):
    expected = frozen_inputs[str(pathlib.Path(p)).lower()]
    assert sha(p) == expected, f'Changed frozen input: {p}'


def decode(runs, shape):
    assert sum(runs) == int(np.prod(shape)) and all(type(x) is int and x>=0 for x in runs)
    return np.repeat(np.arange(len(runs), dtype=np.uint32)%2, runs).reshape(shape).astype(bool)


classes = ['metal','plastic','paper']
class_index = {c:i for i,c in enumerate(classes)}
pixel = {key:np.zeros((4,4),dtype=np.int64) for key in ['native','reference']}
objects = {key:np.zeros((4,4),dtype=np.int64) for key in pixel}
counts = {key:dict(correct=0, wrong_material=0, poor_mask=0, no_candidate=0, empty_images=0, empty_false_positives=0) for key in pixel}
output_rows=[]
frames = [f for f in features['native'] if f['role']=='replay' or f['scene']=='empty']
assert sum(f['role']=='replay' for f in frames)==128
for f in frames:
    image_path=dataset/f['image']
    label_path=dataset/'labels'/pathlib.Path(f['image']).parent.name/pathlib.Path(f['image']).with_suffix('.txt').name
    verify_frozen(image_path); verify_frozen(label_path)
    image=cv2.imread(str(image_path)); assert image.shape==(480,640,3)
    truth=np.zeros((480,640),np.uint8)
    gt=np.full((480,640),3,dtype=np.uint8)
    if f['scene']!='empty':
        v=[float(x) for x in label_path.read_text(encoding='utf-8-sig').split()]
        assert int(v[0])==f['truth_class']
        cv2.fillPoly(truth,[np.rint(np.asarray(v[1:]).reshape(-1,2)*[640,480]).astype(np.int32)],1)
        gt[truth.astype(bool)]=int(v[0])
    picks=[i for i,p in enumerate(f['predictions']) if p['inside_source']>=.8 and p['confidence']>=entry['confidence']]
    selected=picks[0] if picks else None
    mask=None
    predclasses={'native':3,'reference':3}
    iou=0.
    if selected is not None:
        saved=stored[f['image']]['predictions'][selected]
        p=f['predictions'][selected]
        assert abs(saved['confidence']-p['confidence'])<1e-12
        assert saved['class_id']==p['class_id']
        mask=decode(saved['rle'],stored[f['image']]['shape'])
        union=np.logical_or(mask,truth).sum()
        iou=float(np.logical_and(mask,truth).sum()/union) if union else 0.
        assert abs(iou-p['iou'])<1e-12
        predclasses['native']=p['class_id']
        assisted=bank.assist(image,mask,hashlib.sha256(image.tobytes()).hexdigest())
        assert assisted['reference_assisted']
        predclasses['reference']=class_index[assisted['reference_material']]
    for variant,cid in predclasses.items():
        pred=np.full((480,640),3,dtype=np.uint8)
        if mask is not None:pred[mask]=cid
        pixel[variant]+=np.bincount((gt.astype(np.int64)*4+pred).reshape(-1),minlength=16).reshape(4,4)
        if f['scene']=='empty':
            counts[variant]['empty_images']+=1
            counts[variant]['empty_false_positives']+=int(mask is not None)
            continue
        objects[variant][f['truth_class'],cid]+=1
        key='no_candidate' if selected is None else ('wrong_material' if cid!=f['truth_class'] else ('poor_mask' if iou<.5 else 'correct'))
        counts[variant][key]+=1
    output_rows.append(dict(image=f['image'],scene=f['scene'],object_id=f['object_id'],truth=f['truth_class'],
                            selected_prediction_index=selected,predicted_material_ids=predclasses,mask_iou=iou))

quality={}
for variant in pixel:
    cm=pixel[variant]
    cls_iou=[]
    cls_f1=[]
    for c in range(3):
        tp=int(cm[c,c]);fn=int(cm[c,:].sum()-tp);fp=int(cm[:,c].sum()-tp)
        cls_iou.append(tp/(tp+fp+fn))
        oc=objects[variant]
        # A correctly classified mask with IoU<.5 would be an object FP+FN;
        # current selected masks have no poor-mask cases, asserted below.
        otp=int(oc[c,c]);ofn=int(oc[c,:].sum()-otp);ofp=int(oc[:,c].sum()-otp)
        cls_f1.append(2*otp/(2*otp+ofp+ofn))
    assert counts[variant]['poor_mask']==0
    quality[variant]=dict(**counts[variant], eligible_source_images=128,
                          task_candidate_pass_rate=counts[variant]['correct']/128,
                          per_material_IoU=dict(zip(classes,cls_iou)),foreground_3class_mIoU=float(np.mean(cls_iou)),
                          per_material_F1=dict(zip(classes,cls_f1)),macro_F1=float(np.mean(cls_f1)),
                          pixel_confusion_with_background=cm.tolist(),object_confusion_with_reject=objects[variant].tolist())
    assert cm.sum()==len(frames)*640*480
    assert objects[variant][:3,:].sum()==128
    assert sum(counts[variant][k] for k in ['correct','wrong_material','poor_mask','no_candidate'])==128
assert quality['native']['correct']==82
savedref=read(base/'comparison.json')['models']['yoloe26-public-retrained']['variants']['reference']['fixed_replay']['0.1']['overall']
assert quality['reference']['correct']==savedref['correct']==113
training=read(ROOT/'docs/team/reports/2026-10-31-yoloe-public16-results.json')
summary['AI_quality']=dict(scope='offline retrospective known-inventory single-source-object task, 16 objects / 128 replay photos plus 3 empty photos, 64 separate reference/development photos',
                           threshold=.10, inference_source='previously recorded CPU FP32 predictions, not new live GPU FP16 inference',
                           current_weight_and_configuration_hashes_match=True,
                           original_inputs_hash_verified=True, current_reference_bank_and_implementation_used=True,
                           neural_classification_and_reference_material_separate=True,
                           miou_definition='pooled per-material intersection/union over full 640x480 selected-target masks; average over metal/plastic/paper, background excluded from average but retained in FP/FN; misses and wrong classes included',
                           f1_definition='macro average over 3 materials for selected source object; no detection counts as FN, wrong material counts as FP and FN; no poor-mask selected cases at threshold .10',
                           source_instance_detection_mAP_50_95=None, full26_segmentation_mIoU=None,
                           public_7_focus_classes_mask_AP50_95=training['training_complete']['best_focus_mask_ap50_95'],
                           public_validation_images=128, quality=quality,
                           not_independent=True, not_physical_success_rate=True,
                           excludes='unseen objects, current scene calibration, current GPU FP16 full pipeline, physical grasping, 26-fine-class accuracy',
                           runtime_versions={'offline_replay_python':sys.version.split()[0],'opencv':cv2.__version__,'numpy':np.__version__},
                           rows=output_rows)
summary['evidence_sources']={str(p):sha(p) for p in [routefile,buildfile,OUT/'live-samples.json',
    ROOT/'local-only/pc-camera-ab-20261101/board-results.json',base/'yoloe26-public-retrained-features.json',
    base/'comparison.json',pathlib.Path(entry['reference_bank']),ROOT/'pynq_km1_vision/src/inventory_reference.py']}
(OUT/'metrics-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ['communication','live_PL','route_compilation','FPGA_build']},ensure_ascii=False,indent=2))
print(json.dumps(quality,ensure_ascii=False,indent=2))

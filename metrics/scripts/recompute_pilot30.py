"""Offline pilot30 audit from packaged observations. No network or hardware access."""
from pathlib import Path
import csv, hashlib, json, math

root = Path(__file__).resolve().parents[2]
logs = root/'metrics/logs/pilot30'
evidence = root/'metrics/evidence/pilot30'
def j(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def close(a,b): assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-8),(a,b)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

rows=csvrows(logs/'results.csv'); dual=csvrows(logs/'dual_criteria.csv'); stats=j(logs/'statistics.json')
assert len(rows)==len(dual)==30
assert [r['trial_id'] for r in rows]==[f'P{i:03d}' for i in range(1,31)]
assert [r['trial_id'] for r in dual]==[r['trial_id'] for r in rows]
materials={}; times=[]; commands=0; checked_hashes=0
for r,d in zip(rows,dual):
    trial=r['trial_id']; p=evidence/trial.lower()
    assert (logs/r['evidence_directory']).resolve()==p.resolve()
    truth=j(p/'operator-truth.json')
    strict=truth['strict_full_containment']=='pass'
    assert strict==(r['strict_success']=='true')==(d['original_strict_success']=='true')
    rim=truth.get('reason','').startswith('can_rim_resting_on_bin_edge')
    assert (strict or rim)==(d['supplemental_target_bin_acceptance']=='true')
    g=materials.setdefault(r['material'],dict(tasks=0,strict=0,allow_rim=0,motion=0))
    g['tasks']+=1; g['strict']+=strict; g['allow_rim']+=strict or rim
    if r['physical_motion']=='false':
        assert trial in ['P022','P027']
        refusal=j(p/'system-refusal.json')
        assert refusal['real_start_submitted'] is False
        assert refusal['original_result']['status']=='rejected' and refusal['original_result']['preview'] is True
        assert 'physical_motion_attempted' not in refusal['original_result'] or refusal['original_result']['physical_motion_attempted'] is False
        assert not r['latency_mid_s'] and r['no_response']=='true'
        continue
    g['motion']+=1
    s=j(p/'measurement-summary.json')
    frames={x['frame_id']:x for line in (p/'recording/frames.jsonl').read_text(encoding='utf-8').splitlines() if 'frame_id' in (x:=json.loads(line))}
    source=frames[s['input_frame_id']]
    before=frames[s['last_still_frame_id']]; after=frames[s['first_moving_frame_id']]
    assert source['stream_id']==before['stream_id']==after['stream_id']
    lower=before['capture_lower_monotonic_s']-source['capture_upper_monotonic_s']
    upper=after['capture_upper_monotonic_s']-source['capture_lower_monotonic_s']
    midpoint=(lower+upper)/2
    close(lower,r['latency_lower_s']); close(upper,r['latency_upper_s']); close(midpoint,r['latency_mid_s'])
    close(midpoint,s['latency_estimate_s']); close(midpoint,d['latency_mid_s'])
    assert s['active_model']['configuration_sha256']==r['configuration_id']
    before_png=p/'task-evidence'/f"{s['task_id']}.before.png"
    obs=j(before_png.with_suffix('.observation.json'))
    assert sha(before_png)==obs['png_sha256']
    assert obs['decoded_bgr_sha256']==s['input_bgr_sha256']==obs['observation']['source_bgr_sha256']
    assert obs['observation']['source_stream_id']==source['stream_id']
    assert int(obs['observation']['source_frame_id'])==source['frame_id']
    for rel,digest in s['artifact_sha256'].items():
        file=p/rel.replace('\\','/')
        assert file.is_file(),str(file)
        assert sha(file)==digest,rel
        checked_hashes+=1
    report=j(p/'task-evidence'/f"{s['task_id']}.json")
    assert all(x['transport_receipt']['echo_matched'] and x['transport_receipt']['completed_movement_wait'] for x in report['steps'])
    assert len(report['steps'])==s['serial_commands']
    commands+=len(report['steps']); times.append(midpoint)

assert materials=={'metal':dict(tasks=10,strict=2,allow_rim=10,motion=10),'paper':dict(tasks=10,strict=7,allow_rim=7,motion=10),'plastic':dict(tasks=10,strict=8,allow_rim=8,motion=8)}
assert len(times)==28 and commands==616
assert sum(int(r['instrumentation_exceptions']) for r in rows)==1
assert sum(r['harness_version']=='v1_fixed_client_roi' for r in rows)==6
assert sum(r['harness_version']=='v2_registered_source' for r in rows)==24
assert stats['overall']['tasks']==30 and stats['overall']['strict_successes']==17
assert stats['overall']['all_task_physical_response_p99_s'] is None
for group,filtered in [('overall',rows)]+[(m,[r for r in rows if r['material']==m]) for m in materials]:
    vals=sorted(float(r['latency_mid_s']) for r in filtered if r['latency_evidence_valid']=='true')
    expected=stats['overall'] if group=='overall' else stats['by_material'][group]
    assert expected['response']['n']==len(vals)
    for k,q in [('P50',.5),('P95',.95),('P99',.99),('max',1)]: close(vals[math.ceil(q*len(vals))-1],expected['response'][k]['mid_s'])
manual=j(evidence/'p028/manual-motion-review.json')
assert manual['last_clearly_still_frame']==45461 and manual['first_unambiguous_moving_frame']==45463
for name,digest in manual['reviewed_raw_images_sha256'].items(): assert sha(evidence/'p028/recording'/name)==digest
assert sha(evidence/'p028/measurement-summary-automatic-candidate.json')==manual['original_automatic_candidate_sha256']
for item in j(root/'provenance/pilot30_source_manifest.json')['included_files']:
    assert sha(root/item['path'])==item['sha256']; checked_hashes+=1
metrics=csvrows(root/'metrics/metrics.csv')
for key in ['P50','P95','P99','最大值']:
    row=next(r for r in metrics if r['指标名称']=='实际动作条件可见首动 '+key)
    assert abs(float(row['数值'])-stats['overall']['response']['max' if key=='最大值' else key]['mid_s']*1000)<0.000001
print(json.dumps(dict(passed=True,tasks=30,strict=17,allow_recorded_rim=25,by_material=materials,valid_motion_timings=28,refusals=['P022','P027'],nearest_rank_seconds={k:sorted(times)[math.ceil(q*len(times))-1] for k,q in [('P50',.5),('P95',.95),('P99',.99),('max',1)]},serial_echoes_and_waits=commands,hashes_checked=checked_hashes,new_measurements=False,hardware_access=False),ensure_ascii=False,indent=2))

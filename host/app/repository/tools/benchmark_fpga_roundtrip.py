"""CPU versus deployed PL dual-core RPC replay. No serial, motion or bit loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pynq_km1_vision'))
import numpy as np
from src.sorting_io import Statistics,PLStatistics,pack_mask,digest
from src.static_mask_contract import compare_snapshot


def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def measure(values):
    return dict(samples=len(values),median_ms=statistics.median(values),mean_ms=statistics.mean(values),
        p95_ms=float(np.percentile(values,95)),max_ms=max(values),sequential_jobs_per_s=1000/statistics.mean(values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=4)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'local-only') or not 2<=args.repeats<=10:raise ValueError('Invalid output/repeats')
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))['statistics']
    os.environ[config['token_env']]=args.token_file.read_text().strip()
    pl=PLStatistics(config)
    cpu=Statistics()
    source=ROOT/'local-only/rocm-cpu-benchmark/20261029-v8-preparation/yolov8s-gpu_fp16-observations.json'
    rows=json.loads(source.read_text());cases=[]
    for index in (1,2,4,6):
        for obj in rows[index]['instances']:
            runs=obj['mask_rle'];mask=np.repeat(np.arange(len(runs))%2,runs).reshape(480,640).astype(bool)
            changed=np.zeros_like(mask);changed[:,3:]=mask[:,:-3]
            cases.append(dict(name=Path(rows[index]['image']).name+'-'+obj['label'],mask=mask,changed=changed))
    plan=dict(started_at=datetime.now(timezone.utc).isoformat(),assets=config['assets'],session_id=pl.session,
        image_size=[640,480],roles=['geometry','visual'],cases=len(cases),repeats=args.repeats,
        input_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),runner_sha256=digest(Path(__file__).read_bytes()),
        input_note='Recorded validation masks; visual second mask translated three pixels without wrapping.',
        cpu_device='Ryzen AI 9 HX 370 / AI PC',pl_device='PYNQ-Z2 / deployed dual-core',
        scope='PC packbits+base64+JSON+HTTP/network+PS decode+DMA+PL+response; reference/validation timed separately.',
        kernel_only_measured=False,cpu_utilization_measured=False,mechanical_commands_sent=False,
        serial_opened=False,bitstream_loaded=False,test_evaluated=False)
    save(output/'plan.json',plan)
    records=[];counter=0

    def one(role,case,warm=False):
        nonlocal counter
        # A shared board is never benchmarked during an active arm-device lease.
        if (ROOT/'local-only/arm-device-COM9.lock').exists():raise RuntimeError('arm_active_benchmark_stopped')
        counter+=1;job=0x60000000+counter;mask=case['mask'];other=case['changed'];masks=[mask] if role=='geometry' else [mask,other]
        started=time.perf_counter_ns()
        expected=cpu.geometry(mask) if role=='geometry' else cpu.visual(mask,other)
        cpu_ms=(time.perf_counter_ns()-started)/1e6;expected['job_id']=job
        nonce=os.urandom(16).hex();started=time.perf_counter_ns()
        response=pl.request('run',dict(role=role,session_id=pl.session,request_id=nonce,image_size=[640,480],
            job_id=job,masks=[pack_mask(m) for m in masks]))
        rpc_ms=(time.perf_counter_ns()-started)/1e6
        started=time.perf_counter_ns()
        assert response['session_id']==pl.session and response['assets']==config['assets'] and response['request_id']==nonce
        compare_snapshot(response['stats'],expected)
        pixels=mask if role=='geometry' else mask^other
        assert response['output_sha256']==digest(pixels.astype(np.uint8).tobytes())
        validation_ms=(time.perf_counter_ns()-started)/1e6
        if not warm:
            item=dict(role=role,case=case['name'],job_id=job,cpu_ms=cpu_ms,pl_roundtrip_ms=rpc_ms,
                validation_ms=validation_ms,checked_adapter_ms=cpu_ms+rpc_ms+validation_ms,
                input_sha256=[digest(m.astype(np.uint8).tobytes()) for m in masks],
                exact_statistics_match=True,output_hash_match=True)
            records.append(item)
        time.sleep(.03)

    for role in ('geometry','visual'):one(role,cases[0],True)
    with (output/'measurements.jsonl').open('w',encoding='utf-8') as stream:
        for repetition in range(args.repeats):
            for case in cases:
                for role in ('geometry','visual') if repetition%2==0 else ('visual','geometry'):
                    one(role,case);stream.write(json.dumps(records[-1])+'\n');stream.flush()
            print('Completed replay round '+str(repetition+1),flush=True)
    summaries={}
    for role in ('geometry','visual'):
        rows=[r for r in records if r['role']==role]
        baseline=measure([r['cpu_ms'] for r in rows]);remote=measure([r['pl_roundtrip_ms'] for r in rows])
        checked=measure([r['checked_adapter_ms'] for r in rows])
        summaries[role]=dict(cpu=baseline,pl_roundtrip=remote,with_cpu_check=checked,
            median_speedup=baseline['median_ms']/remote['median_ms'],
            batch_speedup=baseline['mean_ms']/remote['mean_ms'],correct=len(rows),total=len(rows))
    report=dict(plan,completed_at=datetime.now(timezone.utc).isoformat(),roles=summaries,
        comparisons=len(records),all_statistics_and_output_hashes_equal=True,
        note='Warm batch1 replay on existing service. No kernel-only timing or whole-system speedup claim.')
    save(output/'result.json',report)
    latest=ROOT/'local-only/fpga-device-benchmark/latest.json';latest.parent.mkdir(parents=True,exist_ok=True)
    save(latest,report)
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()

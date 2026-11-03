"""Bounded loopback access to the camera's board-local evidence endpoint."""
import json
import hashlib
import math
import statistics
from pathlib import Path
import urllib.request

from .station_vision import StationVision,CPUVision

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'pynq_km1_vision/config/station_vision.local.json'
PC_REPORT=ROOT/'local-only/station-vision-deployment/latest-pc-benchmark.json'
BOARD_REPORT=ROOT/'local-only/station-vision-deployment/latest-board-benchmark.json'
BOARD_BINDING=ROOT/'local-only/station-vision-deployment/latest-board-benchmark-binding.json'
ACTIVE=ROOT/'local-only/station-vision-deployment/active.json'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))


def saved_acceleration_summary(config_path=CONFIG,active_path=ACTIVE,board_path=BOARD_REPORT,pc_path=PC_REPORT,binding_path=BOARD_BINDING):
    """Recorded evidence remains visible when capture is disconnected."""
    try:
        config,signature=configured(config_path)
        active=json.loads(Path(active_path).read_text(encoding='utf-8-sig'))
        raw=Path(board_path).read_bytes();board=json.loads(raw.decode('utf-8-sig'))
        binding=json.loads(Path(binding_path).read_text(encoding='utf-8-sig'))
        rows=board['measurements'];count=board['samples']
        if (binding['report_sha256']!=hashlib.sha256(raw).hexdigest() or
            binding['config_sha256']!=signature or binding['assets']!=config['assets'] or
            active['assets']!=config['assets'] or active['config_sha256']!=signature or
            board['assets']!=config['assets'] or board['backend']!='pl_board_local' or
            board['cpu']!='board_ARM_numpy_reference' or
            board['source']!='synthetic_board_startup' or board['all_equal'] is not True or
            board['image_size']!=config['image_size'] or board['threshold']!=config['threshold'] or
            type(count) is not int or count<1 or len(rows)!=count):raise ValueError('unbound_benchmark')
        for row in rows:
            for key in ('cpu_ms','pl_ms'):
                if type(row[key]) not in (int,float) or not math.isfinite(row[key]) or row[key]<=0:
                    raise ValueError('invalid_benchmark_time')
        cpu=statistics.median(r['cpu_ms'] for r in rows);pl=statistics.median(r['pl_ms'] for r in rows)
        startup={k:v for k,v in board.items() if k!='measurements'}
        return dict(available=True,source='saved_board_startup_benchmark',assets=board['assets'],
            config_sha256=signature,samples=count,cpu_median_ms=cpu,pl_median_ms=pl,
            median_speedup=cpu/pl,latency_reduction_percent=(1-pl/cpu)*100,
            pc_reference=pc_reference_summary(startup,pc_path),live_speedup=False,
            whole_system_speedup_measured=False)
    except (OSError,ValueError,TypeError,KeyError):
        return dict(available=False,reason='no_matching_saved_fpga_benchmark')


def pc_reference_summary(startup,path=PC_REPORT):
    try:
        report=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        value=report['cpu_median_ms']
        if (not isinstance(startup,dict) or report.get('board_startup_summary')!=startup or
            report.get('same_input_verified') is not True or report['samples']!=startup['samples'] or
            report.get('assets')!=startup['assets'] or type(value) not in (int,float) or
            not math.isfinite(value) or value<=0):return None
        return dict(cpu=report['cpu'],cpu_median_ms=value,source='same_synthetic_startup_inputs',
            measured_at=report['host_measured_at'],samples=report['samples'],includes_network=False)
    except (OSError,KeyError,TypeError,ValueError):return None


def configured(path=CONFIG):
    value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if value.get('synthetic_fixture') is not False or value.get('backend')!='pl_board_local':
        raise ValueError('reviewed_board_pipeline_required')
    engine=StationVision(CPUVision(),value['regions'],value['image_size'],value['threshold'],
        value['max_gap_s'],value['window_frames'])
    return value,engine.config_sha256


def board_request(route,body=None):
    call=urllib.request.Request('http://127.0.0.1:8780'+route,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Content-Type':'application/json'} if body is not None else {})
    with OPENER.open(call,timeout=3) as reply:
        raw=reply.read(128001)
    if len(raw)>128000:raise ValueError('oversized_station_vision_reply')
    return json.loads(raw)


def station_vision_status(config_path=CONFIG,*,observe_runtime_regions=False):
    try:
        config,signature=configured(config_path);data=board_request('/vision')
        runtime_regions=False
        if observe_runtime_regions and data.get('config_sha256')!=signature:
            # Display-only: relocated execution owns its explicit region
            # transaction. Recompute the observed six-region checksum without
            # granting these coordinates any mechanical qualification.
            if (data.get('threshold')!=config['threshold'] or data.get('window_frames')!=config['window_frames'] or
                set(data.get('regions',{}))!=set(config['regions'])):
                raise ValueError('station_runtime_policy_mismatch')
            engine=StationVision(CPUVision(),data['regions'],config['image_size'],config['threshold'],
                                 config['max_gap_s'],config['window_frames'])
            signature=engine.config_sha256;runtime_regions=True
        if (data.get('contract')!='station-vision-v1' or data.get('backend')!='pl_board_local' or
            data.get('assets')!=config['assets'] or data.get('config_sha256')!=signature or
            data.get('image_size')!=config['image_size']):
            raise ValueError('station_vision_identity_mismatch')
        evidence=data.get('evidence')
        if data.get('available'):
            if (not data.get('camera_connected') or data.get('faulted') or not 0<=data['frame_age_ms']<2000 or
                not isinstance(evidence,dict) or evidence['frame_id']!=data['frame_id'] or
                evidence['baseline_frame_id']!=data['baseline_frame_id'] or evidence['config_sha256']!=signature or
                evidence.get('backend')!='pl_board_local' or evidence.get('automatic_motion_allowed') is not False):
                raise ValueError('station_vision_stale_or_unbound_result')
        data.update(configured=True,automatic_motion_allowed=False,physical_success_verified=False,
            runtime_regions_observed=runtime_regions,region_geometry_qualified=False,
            pc_reference_comparison=pc_reference_summary(data.get('startup_comparison')))
        return data
    except (OSError,KeyError,TypeError,ValueError):
        return dict(available=False,configured=False,reason='board_local_pipeline_not_deployed',
            automatic_motion_allowed=False,physical_success_verified=False)


def bind_station_baseline(body,config_path=CONFIG):
    config,signature=configured(config_path)
    status=station_vision_status(config_path)
    if (not status.get('configured') or status.get('faulted') or not status.get('camera_connected') or
        body.get('stream_id')!=status.get('stream_id') or body.get('config_sha256')!=signature or
        type(body.get('frame_id')) is not int):
        raise ValueError('baseline_identity_mismatch')
    return board_request('/baseline',{key:body[key] for key in ('stream_id','config_sha256','frame_id')})

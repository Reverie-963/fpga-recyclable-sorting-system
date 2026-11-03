"""Voice-specific entry around existing calibrated single-item components.

No changes to identity-bound station sources, ESP32 firmware or robot profiles.
"""
from pathlib import Path
import urllib.request
import json

import cv2

from .km1_voice import check_requested_material
from .sorting_relocated import RelocatedController, RelocatedMotion
from .sorting_relocated_run import (load_binding, integration_blockers, raw_window,
                                    RelocatedGeometry, activate_regions)
from .sorting_relocated_camera import WindowedStationCamera
from .sorting_station_io import StationCamera, StationStatistics
from .sorting_io import Statistics, HTTPYoloe
from .sorting_loop import Journal, need
from .sorting_known import file_hash


class VoiceController(RelocatedController):
    def __init__(self, *args, voice_request, **kwargs):
        self.voice_request = voice_request
        super().__init__(*args, **kwargs)

    def plan(self, row, target):
        # Runs before plan compilation, pending reservation and COM9 opening.
        check_requested_material(self.voice_request, target[0]['label'])
        return super().plan(row, target)


def run_voice_cycle(binding, request, execute=False, ready=False, transport='asrpro_usb_uart0'):
    need(request in ('PAPER', 'METAL', 'PLASTIC', 'AUTO'), 'invalid_voice_sort_request')
    need(transport in ('asrpro_usb_uart0', 'local_dashboard'), 'invalid_sort_request_transport')
    b,c,base,camera,station,registration,scene,files=load_binding(binding)
    blockers=integration_blockers(b,c,base,files)
    health=json.load(urllib.request.urlopen(c['statistics']['url'].rstrip('/')+'/health',timeout=3))
    if 'region_binding_v1' not in health.get('capabilities',[]):
        blockers.append('board_region_binding_update_not_deployed')
    if execute:
        need(ready and not blockers, 'voice_station_not_ready:'+','.join(blockers))
    rows,pose,old_stats=raw_window(c,camera,station,registration,scene)
    geometry=RelocatedGeometry(c,camera,station,registration,scene,pose,rows[-1]['image'])
    geometry.runtime['known']['expected_camera_stream_id']=rows[-1]['observation']['source_stream_id']
    reference=cv2.imread(str(files['empty_image']))
    empty=geometry.warp_empty_reference(reference) if scene.get('empty_bins_confirmed') else geometry.reference.copy()
    if execute:
        activation=activate_regions(old_stats,geometry,c,base)
        stats=StationStatistics(geometry.runtime)
        source=StationCamera(geometry.runtime,HTTPYoloe(c,None),stats)
    else:
        activation=None
        stats=Statistics()
        source=StationCamera(old_stats.c,HTTPYoloe(c,None),old_stats)
    source=WindowedStationCamera(source)
    preview_dir='voice-preview-ledger' if transport=='asrpro_usb_uart0' else 'dashboard-preview-ledger'
    ledger=(base/c['journal_dir']).resolve() if execute else Path(binding).parent/preview_dir
    with Journal(ledger) as journal:
        need(journal.pending() is None, 'unresolved_relocated_trial')
        motion=None
        if execute:
            motion=RelocatedMotion(geometry.runtime,journal,geometry,
                (base/c['known']['device_lease']).resolve(),
                [(base/p).resolve() for p in c['known']['stop_files']],
                (base/c['journal_dir']).resolve()/'pending.json')
        controller=VoiceController(c,source,stats,motion,journal,geometry,empty,base,
            voice_request=request,mode='known_physical' if execute else 'preview',ready=ready,
            motion_profile=c['known'].get('motion_profile','qualified'))
        report=controller.cycle()
        report.update(voice_request=request,voice_transport=transport,
            voice_entry_sha256=file_hash(__file__),
            voice_policy_sha256=file_hash(Path(__file__).with_name('km1_voice.py')),
            integration_binding_sha256=file_hash(binding),region_activation=activation,
            projected_regions=geometry.runtime['statistics']['regions'],
            physical_motion_attempted=execute and bool(report.get('steps')))
        if not execute:
            report['execution_blockers']=blockers
        journal.save(report['task_id']+'.json',report)
        return report

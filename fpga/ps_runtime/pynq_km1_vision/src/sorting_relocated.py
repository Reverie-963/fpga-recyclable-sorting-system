"""Known-item integration: frozen live calibration per cycle, original robot routes."""
import copy
import json
import os
from pathlib import Path
import time

import numpy as np

from .sorting_known import KnownController, GeometryGuardedMotion, SCOPE, known_readiness
from .sorting_relocated_plan import compile_relocated_plan
from .sorting_io import Motion
from .sorting_loop import Controller, need
from .sorting_motion import compact_plan
from .sorting_scene import adjust_exposure_reference
from .static_visual_verification import roi_mask
from openmv_gripper.km1_arm_bridge import validate_packet


class RelocatedMotion(GeometryGuardedMotion):
    def __init__(self, c, journal, geometry, lease, stops, external_pending):
        self.external_pending=Path(external_pending)
        self.owned_task_id=None
        super().__init__(c, journal, geometry.reference, lease, stops)
        self.geometry = geometry

    def check_stop(self):
        super().check_stop()
        if self.external_pending.exists():
            current=json.loads(self.external_pending.read_text(encoding='utf-8'))
            need(getattr(self,'owned_task_id',None) is not None and
                 current.get('task_id')==self.owned_task_id,'existing_physical_trial_unresolved')

    def open(self,plan):
        # Execute uses the SAME locked journal as the old entry. Only this
        # cycle's reserved intent may remain pending while its writes run.
        self.owned_task_id=self.journal.pending()['task_id']
        self.check_stop()
        from serial.tools import list_ports
        identity=self.arm['usb_identity']
        devices=[p.device for p in list_ports.comports() if p.vid==identity['vid'] and p.pid==identity['pid']
                 and (identity.get('serial_number') is None or p.serial_number==identity['serial_number'])]
        need(devices==[self.arm['port']],'known_station_usb_not_unique')
        self.lease.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(self.lease,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);self.lease_owned=True
        try:
            os.write(fd,str(os.getpid()).encode());os.fsync(fd)
        finally:os.close(fd)
        try:
            self.check_stop()  # Repeat after exclusive ownership, before USB reset.
            for _,data,duration in plan:
                need(validate_packet(data,[[500,2500]]*6)==duration,'invalid_known_packet')
            Motion.open(self,plan)
        except BaseException:
            self.close();raise

    def send(self, data, duration, guard=None):
        if getattr(self,'scene_check_mode','per_command')=='before_sequence':
            from .motion_sequence import send_sequence_step
            return send_sequence_step(self,data,duration,guard,Motion.send)
        def checked():
            self.check_stop()
            row=self.camera.raw(self.c['frame_timeout_s'])
            need(row['observation']['source_stream_id']==self.expected_stream(),'motion_camera_stream_changed')
            try:
                barrier=getattr(self,'last_motion_completed_ms',None)
                need(barrier is None or row['observation']['capture_started_ms']>barrier,
                     'motion_guard_frame_precedes_previous_motion_end')
                evidence=self.geometry.check(row['image'])
            except ValueError as exc:
                if getattr(self,'journal',None) is not None:
                    task=self.journal.pending()['task_id']
                    phase='motion_blocked_%02d'%self.guard_index
                    image=self.journal.capture(task,phase,row)
                    self.journal.save(task+'.'+phase+'.json',dict(reason=str(exc),evidence=image,
                        motion_command_written=False,phase='before_motion_uart_write',
                        marker_patch_diagnostics=getattr(self.geometry,'last_patch_attempts',[]),
                        source_frame_id=row['observation']['source_frame_id']))
                raise
            need(0 <= time.monotonic()*1000-row['observation']['capture_started_ms'] <= self.c['max_age_ms'],
                 'relocated_motion_frame_expired')
            task=self.journal.pending()['task_id']
            image=self.journal.capture(task,'motion_%02d'%self.guard_index,row)
            self.journal.save(task+'.motion_%02d.json'%self.guard_index,dict(layout=evidence,evidence=image,
                after_motion_barrier_ms=getattr(self,'last_motion_completed_ms',None),
                capture_started_ms=row['observation']['capture_started_ms']))
            self.guard_index+=1
            if guard:guard()
        # Do not auto-relocalize during a trajectory. Any camera/board change
        # invalidates the frozen plan before the next transport write.
        Motion.send(self,data,duration,checked)

    def wait(self,seconds):
        result=super().wait(seconds)
        # A cached frame captured during the previous move is not evidence of
        # the next stationary prewrite scene. Camera.raw retains its normal
        # deadline and waits for a frame newer than the completed motion wait.
        self.last_motion_completed_ms=time.monotonic()*1000
        self.camera.after(self.last_motion_completed_ms)
        return result


class RelocatedController(KnownController):
    def __init__(self, original, source, statistics, motion, journal, geometry, empty, base,
                 mode='preview',ready=False,**kwargs):
        self.geometry=geometry;self.original=copy.deepcopy(original)
        if mode=='known_physical':
            blockers=known_readiness(original,base,require_empty=False,require_vision_binding=False)
            blockers=[v for v in blockers if not v.startswith('layout_reference:')]
            need(ready and not blockers,'relocated_mechanical_scope_not_ready:'+','.join(blockers))
            need(geometry.scene.get('scene_geometry_confirmed') is True and
                 geometry.scene.get('empty_bins_confirmed') is True,'relocated_scene_not_confirmed')
            need(statistics.backend=='pl_board_local' and
                 statistics.cfg['regions']==geometry.runtime['statistics']['regions'],
                 'relocated_board_regions_not_bound')
        super().__init__(original,source,statistics,motion,journal,geometry.reference,empty,base,
                         mode='simulation' if mode=='simulation' else 'preview',ready=ready,**kwargs)
        self.c=copy.deepcopy(geometry.runtime)
        self.mode=mode

    def check_layout(self,image):
        return self.geometry.check(image)

    def plan(self,row,target):
        obj,mask=target;p=self.c['profiles'][obj['label']]
        region=roi_mask(p['bin_roi'],self.c['image_size'])
        need(not any((m&region).any() for m in row['masks']),'known_destination_not_empty_semantically')
        need(self.empty_reference is not None,'known_empty_reference_required')
        empty,_=adjust_exposure_reference(self.empty_reference,row['image'],dict(self.c['layout'],image_size=self.c['image_size']))
        difference=np.abs(row['image'].astype(np.int16)-empty.astype(np.int16)).sum(axis=2)
        need(float((difference[region]>self.c['rgb_difference_threshold']).mean())<=self.c['max_scene_changed_fraction'],
             'known_destination_rgb_not_empty')
        choice,plan=compile_relocated_plan(self.c,obj['label'],mask,self.stats.geometry(mask),
                                       pixel_to_card=lambda pts:self.geometry.map(obj['label'],pts))
        choice['station_calibration']=dict(calibration_sha256=self.geometry.signature,
            source_frame_id=row['observation']['source_frame_id'],source_stream_id=row['observation']['source_stream_id'],
            bgr_sha256=row['bgr_sha256'],height_mm=p['object_height_mm'],
            robot_point_domains_preserved=True,old_image_homography_used=False)
        if self.motion_profile=='compact':plan,choice['motion_optimization']=compact_plan(self.c,plan)
        return choice,plan

    def cycle(self):
        report=Controller.cycle(self)
        report.update(operation_scope=SCOPE,single_cycle_only=True,production_quality_claimed=False,
            unattended_accepted=False,in_grip_vision_claimed=False,motion_profile=self.motion_profile,
            station_calibration_sha256=self.geometry.signature,automatic_camera_update_during_motion=False)
        if self.mode=='preview':report['execution_blockers']=known_readiness(self.original,self.config_base)
        self.journal.save(report['task_id']+'.json',report)
        return report

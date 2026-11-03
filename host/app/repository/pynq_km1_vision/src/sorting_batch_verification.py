"""One final scene verdict for completed routes, with every local verdict kept."""
import numpy as np
from .sorting_loop import need
from .sorting_instances import associate
from .static_visual_verification import roi_mask


def verify_batch_final(controllers,journal,batch_id):
    need(controllers,'batch_final_requires_completed_motion')
    if getattr(controllers[-1],'placement_verification',{}).get('mode')=='bin_occupied':
        return verify_batch_bin_occupied(controllers,journal,batch_id)
    first,last=controllers[0],controllers[-1]
    before=first.batch_context['before']
    contexts=[controller.batch_context for controller in controllers]
    attempted={ctx['track_id'] for ctx in contexts}
    need(len(attempted)==len(contexts),'batch_target_repeated')
    expected=[(o,m) for track,o,m in first.tracker.initial if track not in attempted]
    old_bins=first.outside_objects(before)
    rows=[];result=dict(status='batch_final_verification_failed',frames=[],evidence=[],
        attempted_count=len(contexts),physical_success_verified=False,
        scope='whole_batch_semantic_and_rgb; board_frames_identity_only',automatic_motion_retry=False)
    try:
        last.verification_frames(last.clock()*1000,rows)
        gain_windows=[[] for _ in contexts]
        target_windows=[[] for _ in contexts]
        for row in rows:
            current=last.source_objects(row)
            associate(expected,current,last.c['min_scene_iou'])
            outside=last.outside_objects(row)
            kept=associate(old_bins,outside,last.c['min_scene_iou'],allow_extra=True)
            new=[pair for j,pair in enumerate(outside) if j not in kept]
            need(len(new)==len(contexts),'batch_new_bin_instance_count')
            changed=last.changed_pixels(before,row)
            used=set();placements=[]
            source_union=np.logical_or.reduce([m for _,m in current]) if current else np.zeros_like(changed)
            visited={ctx['choice']['bin_id'] for ctx in contexts}
            for index,ctx in enumerate(contexts):
                obj,selected=ctx['target'];choice=ctx['choice']
                destination=roi_mask(choice['slot_roi'],last.c['image_size'])
                profile=last.c['profiles'][obj['label']]
                matches=[j for j,(o,m) in enumerate(new) if j not in used and
                    o['label']==obj['label'] and o['confidence']>=last.c['min_confidence'] and
                    profile['min_area_px']<=int(m.sum())<=profile['max_area_px'] and
                    int((m&destination).sum())>=m.sum()*.95]
                need(len(matches)==1,'batch_wrong_class_or_landing_slot')
                j=matches[0];used.add(j);placed,mask=new[j]
                erased=last.stats.visual(selected,source_union)
                need(erased['erased']/int(selected.sum())>=last.c['min_erased_fraction'],
                     'batch_selected_target_still_in_source')
                gain=mask&destination&changed
                need(int(gain.sum())>=last.c['min_bin_gain_px'],'batch_rgb_gain_missing')
                gain_windows[index].append(gain);target_windows[index].append(mask)
                placements.append(dict(track_id=ctx['track_id'],label=placed['label'],
                    confidence=placed['confidence'],slot_id=choice['slot_id'],gain_px=int(gain.sum())))
            for _,mask in expected+old_bins:
                need(float(changed[mask].mean())<=last.c['max_scene_changed_fraction'],
                     'batch_untargeted_object_changed')
            for p in last.c['profiles'].values():
                if p['bin_id'] not in visited:
                    region=roi_mask(p['bin_roi'],last.c['image_size'])
                    need(float(changed[region].mean())<=last.c['max_other_bin_changed_fraction'],
                         'batch_unvisited_bin_changed')
            for rect in last.c['layout']['static_rois']:
                need(float(changed[roi_mask(rect,last.c['image_size'])].mean())<=last.c['max_scene_changed_fraction'],
                     'batch_static_background_changed')
            board=None
            if last.mode=='known_physical':
                original=last.stats.c
                last.stats.c=dict(original,max_age_ms=last.policy.get('verification_max_age_ms',5000))
                try:
                    evidence=last.stats.evidence(row)
                    board=dict(frame_id=evidence['frame_id'],baseline_frame_id=evidence['baseline_frame_id'],
                               scope='last_item_baseline; not whole_batch_difference')
                finally:last.stats.c=original
            result['frames'].append(dict(frame_id=row['observation']['source_frame_id'],
                placements=placements,board_frame=board))
        for gains,targets in zip(gain_windows,target_windows):
            need(int(np.logical_and.reduce(gains).sum())>=last.c['min_bin_gain_px'],
                 'batch_rgb_gain_not_persistent')
            need(np.logical_and.reduce(targets).sum()/np.logical_or.reduce(targets).sum()>=
                 last.c['min_persistent_fraction'],'batch_final_target_unstable')
        age=last.policy.get('verification_max_age_ms',5000)
        need(all(0<=last.clock()*1000-r['observation']['capture_started_ms']<=age for r in rows),
             'batch_final_frames_expired')
        result['status']='batch_visual_evidence_pass'
    except Exception as exc:
        result['reason']=type(exc).__name__+':'+str(exc)
    finally:
        for i,row in enumerate(rows):result['evidence'].append(journal.capture(batch_id,'final-'+str(i),row))
        journal.save(batch_id+'.batch-final-verification.json',result)
    return result


def verify_batch_bin_occupied(controllers,journal,batch_id):
    """One fresh final frame; one occupancy verdict per visited destination."""
    first,last=controllers[0],controllers[-1]
    need(last.policy.get('batch_final_operator_statement') and
         last.policy.get('skip_per_item_visual_verification') is True,'batch_final_occupancy_not_authorized')
    rows=[];result=dict(status='batch_final_verification_failed',mode='bin_occupied',
        scope='visited_bin_occupancy_only',attempted_count=len(controllers),frames=[],bins=[],evidence=[],
        physical_success_verified=False,class_verified=False,new_item_verified=False,source_removal_verified=False,
        automatic_motion_retry=False)
    try:
        tracks=[controller.batch_context['track_id'] for controller in controllers]
        need(len(tracks)==len(set(tracks)),'batch_target_repeated')
        last.verification_frames(last.clock()*1000,rows)
        need(len(rows)==1,'batch_final_requires_one_frame')
        visited={}
        for controller in controllers:
            context=controller.batch_context
            visited.setdefault(context['choice']['bin_id'],context)
        for bin_id,context in visited.items():
            label=context['target'][0]['label']
            verdict=last.verify_bin_occupied(first.batch_context['before'],label,context['choice'],rows[0])
            result['bins'].append(dict(verdict,bin_id=bin_id))
        age=last.policy.get('verification_max_age_ms',5000)
        need(0<=last.clock()*1000-rows[0]['observation']['capture_started_ms']<=age,'batch_final_frames_expired')
        result['frames']=[dict(frame_id=rows[0]['observation']['source_frame_id'])]
        result['status']='batch_visual_evidence_pass'
    except Exception as exc:
        result['reason']=type(exc).__name__+':'+str(exc)
    finally:
        for i,row in enumerate(rows):result['evidence'].append(journal.capture(batch_id,'final-'+str(i),row))
        journal.save(batch_id+'.batch-final-verification.json',result)
    return result

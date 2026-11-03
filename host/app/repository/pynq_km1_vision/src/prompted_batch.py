"""One bounded manual request for the initial registered scene; no motion retry."""
import copy
import json
from .sorting_loop import need
from .sorting_prompted_motion import PromptedMotionController
from .sorting_unified_run import run_batch


def batch_settings(options):
    enabled=options.get('automatic_batch_enabled',False)
    need(type(enabled) is bool,'invalid_prompted_batch_option')
    if not enabled:return None
    need(isinstance(options.get('batch_final_operator_statement'),str) and
         bool(options['batch_final_operator_statement'].strip()),'batch_final_operator_statement_required')
    count=options.get('batch_max_items',6);seconds=options.get('batch_max_seconds',1200)
    need(type(count) is int and 1<=count<=6 and type(seconds) in (int,float) and 0<seconds<=1200,
         'invalid_prompted_batch_limits')
    return dict(max_items=count,max_seconds=seconds,operator_statement=options['batch_final_operator_statement'])


class PromptedBatchController(PromptedMotionController):
    def select(self,row):
        initial=not self.tracker.initialized
        selected,rejected=super().select(row)
        if initial:
            need(not rejected,'prompted_batch_initial_route_rejected:'+str(rejected))
        return selected,rejected


def freeze_batch_plan(controller,max_items):
    """Compile the initial scene in full, reserving checked landing slots."""
    before=controller.observe()[-1]
    _,rejected=controller.select(before)
    need(not rejected,'prompted_batch_initial_route_rejected:'+str(rejected))
    ranked=controller.ranked_plans
    need(len(ranked)<=max_items,'prompted_batch_initial_scene_exceeds_limit')
    tracks={track:(obj,mask) for track,obj,mask in controller.tracker.live}
    planned=[];reserved=set()
    original_slots=copy.deepcopy(controller.policy['bin_slots'])
    try:
        for index,(choice,plan) in enumerate(ranked):
            track=choice['track_id'];target=tracks[track]
            # Explicitly disabled destination occupancy allows the existing
            # fixed release route to be shared. Otherwise reserve each slot.
            shared=choice.get('destination_empty_check_enabled') is False
            key=(choice['bin_id'],choice['slot_id'])
            if not shared and key in reserved:
                label=choice['label']
                controller.policy['bin_slots'][label]=[s for s in original_slots[label]
                    if (choice['bin_id'],s['slot_id']) not in reserved]
                choice,plan=controller.candidate(before,target,track)
                key=(choice['bin_id'],choice['slot_id'])
            need(shared or key not in reserved,'prompted_batch_landing_slot_already_reserved')
            reserved.add(key)
            planned.append(dict(before=before,index=index,pickup_source_roi=copy.deepcopy(controller.c['pickup_source_roi']),
                selected=(track,target,copy.deepcopy(choice),copy.deepcopy(plan))))
    finally:
        controller.policy['bin_slots']=original_slots
    controller.candidate_order=[copy.deepcopy(p['selected'][2]) for p in planned]
    return planned


def run_registered_batch(factory,journal,settings,owner_check,clock):
    manifest=run_batch(factory,journal,settings['max_items'],settings['max_seconds'],owner_check,clock=clock,
                       freeze_plan=freeze_batch_plan)
    manifest['task_id']=manifest['batch_id']
    manifest['batch_status']=manifest['status']
    manifest['operation_scope']='supervised-initial-registered-scene-batch-v1'
    manifest['batch_verification_mode']='end_of_batch'
    manifest['batch_planning_mode']='initial_scene_all_routes_before_motion'
    manifest['skip_per_item_visual_verification']=True
    manifest['physical_motion_attempted']=any(item.get('physical_motion_attempted',False) for item in manifest['items'])
    manifest['physical_validation_completed']=False
    manifest['mode']='known_physical' if manifest['physical_motion_attempted'] else 'preview'
    manifest['steps']=[];manifest['plan']=[];manifest['evidence']=[]
    for index,item in enumerate(manifest['items']):
        path=journal.root/(item['task_id']+'.json')
        child=json.loads(path.read_text(encoding='utf-8'))
        manifest['steps'].extend(dict(step,batch_item_index=index) for step in child.get('steps',[]))
        manifest['plan'].extend(dict(step,batch_item_index=index) for step in child.get('plan',[]))
        manifest['evidence'].extend(child.get('evidence',[]))
    manifest['verification']=manifest.get('final_verification',{})
    completed=(manifest['status'] in ('completed','selection_complete') and
               manifest.get('final_verification',{}).get('status')=='batch_visual_evidence_pass')
    if completed:manifest['status']='completed_visual_evidence'
    elif (manifest['physical_motion_attempted'] or manifest['motion_completed_count'] or
          manifest['status']=='needs_review'):
        manifest['status']='needs_review'
        journal.save('batch-pending.json',manifest)
    elif manifest['status'] not in ('empty_scene','selection_complete'):manifest['status']='rejected'
    journal.save(manifest['task_id']+'.json',manifest)
    return manifest

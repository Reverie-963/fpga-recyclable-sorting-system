"""Explicit supervised grasp-plane requests, separate from old qualification."""
import copy
from .sorting_loop import need,finite


def requested_grasp_planes(config,policy,options,scope):
    overrides=options.get('requested_grasp_z_mm',{})
    extension=options.get('requested_jaw_gap_extension')
    if not overrides and extension is None:return config,policy,None
    need(scope.get('basis')=='operator_supervised_trial' and options.get('grasp_override_operator_statement'),
         'grasp_override_requires_supervised_trial')
    need(isinstance(overrides,dict) and set(overrides).issubset(config['profiles']), 'invalid_grasp_override_material')
    c,p=copy.deepcopy(config),copy.deepcopy(policy);changes={}
    for label,z in overrides.items():
        profile,model=c['profiles'][label],p['models'][label]
        need(finite(z) and 0<=z<profile['hover_z_mm'],'invalid_requested_grasp_z')
        old=profile['grasp_z_mm']
        need(model['grasp_z_mm']==old and all(chain[-1][0]==old and chain[-2][0]>z for chain in model['source_chains']),
             'grasp_override_model_chain_mismatch')
        profile['grasp_z_mm']=model['grasp_z_mm']=z
        for chain in model['source_chains']:chain[-1][0]=z
        changes[label]=dict(previous_grasp_z_mm=old,requested_grasp_z_mm=z)
    if extension is not None:
        need(isinstance(extension,dict) and extension.get('kind')=='operator-supervised-linear-extension-v1'
             and extension.get('operator_statement') and extension.get('physical_validation_completed') is False
             and finite(extension.get('min_gap_mm')) and extension['min_gap_mm']>=0,'invalid_requested_jaw_extension')
        p['jaw_gap_extension']=copy.deepcopy(extension)
    return c,p,dict(kind='operator-requested-grasp-plane-v1',changes=changes,jaw_gap_extension=copy.deepcopy(extension),
        operator_statement=options['grasp_override_operator_statement'],physical_validation_completed=False,
        original_qualification_preserved=True,full_route_validation_required=True)

"""Validate shared CPU/PL and execution parameters before touching devices."""
import math

def validate(c):
    v=c['vision'];cam=c['camera'];a=c['arm']
    def integer(x,low,high):return isinstance(x,int) and not isinstance(x,bool) and low<=x<=high
    if not all(integer(cam[k],1,1920) for k in ('width','height')): raise ValueError('Invalid frame size')
    if cam['width']*cam['height']*4 >= 2**23: raise ValueError('Frame exceeds DMA transfer size')
    if len(v['background_bgr'])!=3 or not all(integer(n,0,255) for n in v['background_bgr']): raise ValueError('Invalid background')
    if not integer(v['threshold'],0,765): raise ValueError('Threshold must be 0..765')
    if not integer(v['stable_frames'],1,1000) or not 0<=v['stable_px']<1000: raise ValueError('Invalid stability setting')
    if not 0<v['min_area']<=v['max_area']<=cam['width']*cam['height']: raise ValueError('Invalid blob area')
    if not math.isfinite(v['min_margin']) or v['min_margin']<0: raise ValueError('Invalid margin')
    if not integer(a['move_ms'],300,5000) or not 0<=a['settle_s']<=10: raise ValueError('Invalid move timing')
    if len(a['pick_bounds_mm'])!=4: raise ValueError('Invalid pick bounds')
    xmin,xmax,ymin,ymax=a['pick_bounds_mm']
    if not all(math.isfinite(n) for n in a['pick_bounds_mm']) or not (xmin<xmax and 0<=ymin<ymax): raise ValueError('Invalid pick bounds')
    if not isinstance(a['commissioned'],bool): raise ValueError('commissioned must be boolean')
    if 'base_yaw_calibration' in a:
        from .arm import validate_base_yaw_calibration
        validate_base_yaw_calibration(a['base_yaw_calibration'])
    c.setdefault('inference',{'mode':'local','score_kind':'ridge_margin','max_result_age_s':1.5,'timeout_s':1.0})
    c.setdefault('telemetry',{'preview_interval_s':2,'flush_interval_s':1})
    c.setdefault('verification',{'bin_rois':{},'min_bin_gain_px':100})
    ic=c['inference']
    if ic['mode'] not in ('local','remote') or v['backend'] not in ('cpu','pl'):raise ValueError('Invalid execution mode')
    if ic['score_kind'] not in ('ridge_margin','probability_margin','detection_confidence'):raise ValueError('Invalid score semantics')
    if ic['mode']=='local' and ic['score_kind']!='ridge_margin':raise ValueError('Local board mode uses ridge only')
    if not all(isinstance(ic[k],(int,float)) and math.isfinite(ic[k]) for k in ('timeout_s','max_result_age_s')):raise ValueError('Nonfinite inference limits')
    if not 0<ic['timeout_s']<=ic['max_result_age_s']<=10:raise ValueError('Invalid inference time limits')
    for key,value in c['telemetry'].items():
        if key not in ('preview_interval_s','flush_interval_s') or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Invalid telemetry setting')
    if v.get('pl_kernel','threshold') not in ('threshold','open'):raise ValueError('Invalid PL kernel')
    for roi in c['verification'].get('bin_rois',{}).values():
        if len(roi)!=4 or not all(type(x)==int for x in roi):raise ValueError('Invalid verification ROI')
        x,y,w,h=roi
        if not (w>0 and h>0 and 0<=x<x+w<=cam['width'] and 0<=y<y+h<=cam['height']):raise ValueError('Verification ROI outside frame')
    return c

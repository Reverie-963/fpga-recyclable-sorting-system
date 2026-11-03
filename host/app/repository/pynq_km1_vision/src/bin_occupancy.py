"""Small RGB occupancy check, independent of detector labels and masks."""
import cv2
import numpy as np


def verification_options(policy):
    value=policy.get('placement_verification',{'mode':'semantic_rgb'})
    if not isinstance(value,dict) or value.get('mode') not in ('semantic_rgb','bin_occupied'):
        raise ValueError('invalid_placement_verification_mode')
    if value['mode']=='semantic_rgb':
        return dict(mode='semantic_rgb')
    delta=value.get('rgb_difference_threshold',45)
    area=value.get('min_component_px',64)
    confidence=value.get('min_detection_confidence',.2)
    if type(delta) is not int or not 1<=delta<=765:
        raise ValueError('invalid_bin_occupancy_rgb_threshold')
    if type(area) is not int or area<1:
        raise ValueError('invalid_bin_occupancy_component_area')
    if type(confidence) not in (int,float) or not 0<=confidence<=1:
        raise ValueError('invalid_bin_occupancy_detection_confidence')
    return dict(mode='bin_occupied',rgb_difference_threshold=delta,min_component_px=area,
                min_detection_confidence=confidence)


def bin_occupancy(image,empty_reference,roi,options):
    """Find visible connected changes inside one bin against an adjusted background."""
    if (not isinstance(image,np.ndarray) or not isinstance(empty_reference,np.ndarray) or
            image.dtype!=np.uint8 or empty_reference.dtype!=np.uint8 or
            image.ndim!=3 or image.shape[2]!=3 or image.shape!=empty_reference.shape):
        raise ValueError('invalid_bin_occupancy_images')
    if (not isinstance(roi,(list,tuple)) or len(roi)!=4 or any(type(v) is not int for v in roi)):
        raise ValueError('invalid_bin_occupancy_roi')
    x0,y0,x1,y1=roi
    if not (0<=x0<x1<=image.shape[1] and 0<=y0<y1<=image.shape[0]):
        raise ValueError('invalid_bin_occupancy_roi')
    opts=verification_options({'placement_verification':options})
    if opts['mode']!='bin_occupied':
        raise ValueError('bin_occupancy_mode_required')
    difference=np.abs(image[y0:y1,x0:x1].astype(np.int16)-
                      empty_reference[y0:y1,x0:x1].astype(np.int16)).sum(2)
    foreground=(difference>opts['rgb_difference_threshold']).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(foreground,connectivity=8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0
    return dict(occupied=largest>=opts['min_component_px'],bin_roi=list(roi),
                changed_px=int(foreground.sum()),largest_component_px=largest,**opts)

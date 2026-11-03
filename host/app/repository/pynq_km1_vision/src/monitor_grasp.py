"""Display-only points derived from masks of the exact displayed image."""
import numpy as np
import cv2
from time import perf_counter_ns


def display_candidates(observation):
    started=perf_counter_ns()
    decode_ns=point_ns=0
    decoded=0
    result=dict(observation)
    result.pop("display_geometry_timing",None)
    result['display_grasp_points']=[]
    size=result.get('image_size')
    if (not isinstance(size,list) or len(size)!=2 or any(type(v) is not int for v in size) or
        not 1<=size[0]<=1920 or not 1<=size[1]<=1080):return result
    width,height=size
    instances=result.get('instances',[])
    if not isinstance(instances,list):return result
    for item in instances[:20]:
        if not isinstance(item,dict):continue
        runs=item.get('mask_rle')
        if (not isinstance(runs,list) or not runs or any(type(v) is not int or v<0 for v in runs) or
            sum(runs)!=width*height):continue
        stage=perf_counter_ns()
        mask=np.repeat(np.arange(len(runs),dtype=np.uint32)%2,runs).reshape(height,width).astype(np.uint8)
        decode_ns+=perf_counter_ns()-stage
        decoded+=1
        stage=perf_counter_ns()
        ys,xs=np.nonzero(mask)
        if not len(xs):
            point_ns+=perf_counter_ns()-stage
            continue
        distance=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,5)[1:-1,1:-1]
        yy,xx=np.where(distance>=float(distance.max())*.8)
        index=int(np.argmin((xx-xs.mean())**2+(yy-ys.mean())**2))
        result['display_grasp_points'].append(dict(instance_id=item.get('instance_id'),label=item.get('label'),
            pixel=[int(xx[index]),int(yy[index])],kind='image_candidate',method='mask_interior_cpu',
            calibrated=False,control_authorized=False))
        point_ns+=perf_counter_ns()-stage
    result['display_geometry_timing']=dict(
        schema_version='monitor-display-geometry-timing-v1', backend='pc_cpu_opencv_numpy',
        scope='display_candidates_all_instances', elapsed_ms=(perf_counter_ns()-started)/1e6,
        mask_decode_ms=decode_ns/1e6, point_compute_ms=point_ns/1e6,
        decoded_masks=decoded, point_count=len(result['display_grasp_points']),
        frame_id=result.get('frame_id'), source_frame_id=result.get('source_frame_id'),
        source_stream_id=result.get('source_stream_id'), model_version=result.get('model_version'),
        runtime_id=result.get('runtime_id'), calibrated=False, control_authorized=False)
    return result

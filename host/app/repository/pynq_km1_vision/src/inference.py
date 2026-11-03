"""CPU ridge and replaceable ONNX execution providers; no board imports."""
import hashlib,json,math
from pathlib import Path
import cv2,numpy as np
from .vision import Classifier

def crop_tensor(crop,size):
    if crop is None or crop.size==0:raise ValueError('Empty crop')
    image=cv2.resize(crop,(size,size),interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.

class Predictor:
    def __init__(self,config):
        self.kind=config['kind'];path=Path(config['model'])
        self.model_id=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.kind=='ridge':
            self.model=Classifier(path);self.providers=['numpy_cpu'];self.metric='ridge_margin'
            self.labels=self.model.labels
        elif self.kind=='onnx':
            import onnxruntime as ort
            metadata_bytes=Path(config['metadata']).read_bytes()
            self.model_id=hashlib.sha256(path.read_bytes()+b'\0'+metadata_bytes).hexdigest()
            meta=json.loads(metadata_bytes);self.size=meta['image_size']
            self.labels=meta['labels'];self.metric='probability_margin'
            if meta.get('preprocess')!='rgb_nchw_0_1' or not isinstance(self.size,int) or not 8<=self.size<=512:raise ValueError('Unsupported model metadata')
            requested=config.get('providers',['CPUExecutionProvider'])
            if not requested or any(p not in ort.get_available_providers() for p in requested):raise RuntimeError('Requested provider unavailable: '+str(requested))
            if 'CUDAExecutionProvider' in requested:
                # Load CUDA/cuDNN DLLs before session creation (Windows GPU wheels).
                if hasattr(ort, 'preload_dlls'):
                    ort.preload_dlls()
                else:
                    raise RuntimeError('CUDA deployment requires onnxruntime-gpu >= 1.21')
            options=ort.SessionOptions();options.enable_profiling=config.get('profile',False)
            options.log_severity_level=1 if config.get('verbose',False) else 3
            self.model=ort.InferenceSession(str(path),sess_options=options,providers=requested,provider_options=config.get('provider_options',[{} for _ in requested]))
            self.model.disable_fallback()
            self.providers=self.model.get_providers()
            if requested[0] not in self.providers:raise RuntimeError('Requested provider failed to initialize')
            inputs=self.model.get_inputs()
            if len(inputs)!=1 or inputs[0].type!='tensor(float)':raise ValueError('Expected one float32 model input')
            self.input=inputs[0].name
            # Warm up compilation and verify output before accepting live requests.
            self.predict(np.zeros((self.size,self.size,3),np.uint8))
        else:raise ValueError('Unknown model kind')
        if len(self.labels)<2 or len(set(self.labels))!=len(self.labels) or not all(isinstance(x,str) for x in self.labels):raise ValueError('Need >=2 distinct string labels')
    def predict(self,crop):
        if self.kind=='ridge':label,margin=self.model.predict(crop)
        else:
            logits=np.asarray(self.model.run(None,{self.input:crop_tensor(crop,self.size)})[0]).reshape(-1)
            if len(logits)!=len(self.labels) or not np.isfinite(logits).all():raise ValueError('Invalid model output')
            probs=np.exp(logits-logits.max());probs/=probs.sum();order=np.argsort(probs)
            label=self.labels[order[-1]];margin=float(probs[order[-1]]-probs[order[-2]])
        if not math.isfinite(margin) or margin<0:raise ValueError('Invalid classifier margin')
        return {'label':label,'margin':float(margin),'score_kind':self.metric,'model_id':self.model_id,'providers':self.providers}
    def close(self):
        if self.kind=='onnx' and self.model.get_session_options().enable_profiling:
            return self.model.end_profiling()

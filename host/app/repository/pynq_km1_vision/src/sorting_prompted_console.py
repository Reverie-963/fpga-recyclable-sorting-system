"""Existing station console hosting the 24-prompt preview and empty-scene records."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error
from .monitor_grasp import display_candidates
from .sorting_monitor_server import make_server
from .monitor_operation import default_operations
from .monitor_control import ROOT


class PreviewOperations:
    """Keep historical telemetry, but do not dispatch old-model motion here."""
    def __init__(self):
        self.original=default_operations(ROOT)
        self.backend=None

    def status(self):
        state=self.original.status()
        state.update(configured=False,blockers=['prompted_model_preview_only'],
            home_supported=False,reset_supported=False)
        state['voice_service'].update(supported=False,connected=False)
        return state

    def command(self, payload):
        raise ValueError('prompted_model_preview_only')

    def close(self):
        self.original.close()


def console_page(class_count=24, trained=False):
    page=Path(__file__).with_name('monitor_assets').joinpath('dashboard.html').read_text(encoding='utf-8')
    page=page.replace('<title>可回收物分拣 · 工位监视器</title>',
                      '<title>可回收物中控台 · YOLOE 24 类</title>')
    page=page.replace('<p class="control-step"><span>02</span>',
        '<label for="operation-height">本次物体高度（mm，物体顶部相对桌面）</label>'
        '<input id="operation-height" type="number" min="1" max="120" step="any" required style="width:150px" aria-describedby="operation-height-note">'
        '<p id="operation-height-note" class="caption" aria-live="polite">填写后直接用于下一次所选类别抓取的位置换算。</p>'
        '<p class="control-step"><span>02</span>')
    page=page.replace('目标箱为空、','')
    script="""<script>
window.addEventListener('load',()=>{
 const nav=document.querySelector('.view-tabs');
 const tab=document.createElement('button');tab.textContent='YOLOE · 24 类与空背景';
 tab.id='tab-prompted';tab.setAttribute('role','tab');tab.setAttribute('aria-controls','view-prompted');
 nav.prepend(tab);
 const panel=document.createElement('section');panel.id='view-prompted';panel.setAttribute('role','tabpanel');
 panel.setAttribute('aria-labelledby','tab-prompted');
 const note=document.createElement('p');note.className='caption';
 note.textContent='已完成现有照片调参；空背景只记录，暂不训练。运动窗口提供新模型单件试验，类别与现场条件检查通过后才能抓取。';
 const motionButton=document.createElement('button');motionButton.textContent='打开运动对接窗口';
 motionButton.onclick=()=>document.getElementById('tab-live').click();
 const frame=document.createElement('iframe');frame.src='/prompted/';frame.title='24 类识别、参数与空背景记录';
 frame.style='width:100%;height:calc(100vh - 220px);min-height:750px;border:0;border-radius:12px';
 panel.append(note,motionButton,frame);nav.after(panel);
 const keys=['live','tasks','performance','upload'];
 function show(){for(const key of keys){document.getElementById('view-'+key).hidden=true;document.getElementById('tab-'+key).setAttribute('aria-selected','false')}
 panel.hidden=false;tab.setAttribute('aria-selected','true');document.getElementById('activity-banner').hidden=true;}
 tab.onclick=show;
 for(const key of keys)document.getElementById('tab-'+key).addEventListener('click',()=>{panel.hidden=true;tab.setAttribute('aria-selected','false');document.getElementById('activity-banner').hidden=false;});
 document.getElementById('tab-live').textContent='运动对接窗口';show();
});</script>"""
    page=page.replace('</body>',script+'</body>')
    if trained:
        page=page.replace('24 类',str(class_count)+' 类微调')
        page=page.replace('已完成现有照片调参；空背景只记录，暂不训练。',
            '已加载网上样本训练的26类候选；工位过滤保留全部原始轮廓。')
    return page


def create_console(port=8810, upstream='http://127.0.0.1:8811', operations=None):
    server=make_server(port,upstream=upstream,operations=operations if operations is not None else PreviewOperations())
    original=server.RequestHandlerClass
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def recognition(route, payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        request=urllib.request.Request(upstream+route,data=data,
            headers={'Content-Type':'application/json'} if data is not None else {})
        with opener.open(request,timeout=120) as response:return json.load(response)

    def health():
        value=recognition('/health');settings=value['settings']
        return dict(value,model_name=value.get('model_name','YOLOE · TACO 24 类'),model_loaded=True,
            model_family='yoloe',model_sha256=value.get('model_sha256') or value['model_hashes']['yoloe-11m-seg.pt'],
            runtime_id=settings['settings_sha256'],live_enabled=value['enabled'],
            live_available=True,default_prompt_mode='text',prompt_modes=['text'],
            confidence_threshold=settings['display_confidence'])

    def observation(value):
        result=dict(value,model_version=value['model_sha256'],runtime_id=value['settings_sha256'],
                    frame_id=value.get('source_frame_id',str(value['generated_at'])))
        result['instances']=[dict(row,label=row['material']) for row in value['instances'] if row['visible']]
        return display_candidates(result)

    class Handler(original):
        def proxy_prompted(self):
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            route=self.path[len('/prompted'):]
            if not (route in ('/','/health','/policy','/vocabulary','/camera.jpg','/live-result',
                              '/configure','/detect','/plan','/backgrounds') or
                    (route.startswith('/background/') and route.endswith('.jpg'))):
                return self.reply(404,dict(error='not_found'))
            data=None
            if self.command=='POST':
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=18_000_000 or self.headers.get('Transfer-Encoding'):
                    return self.reply(400,dict(error='invalid_request'))
                data=self.rfile.read(length)
                if len(data)!=length:return self.reply(400,dict(error='incomplete_request'))
            request=urllib.request.Request(upstream+route,data=data,
                headers={'Content-Type':'application/json'} if data is not None else {})
            try:
                try:response=opener.open(request,timeout=110 if route=='/plan' else 120)
                except urllib.error.HTTPError as exc:response=exc
                with response:
                    raw=response.read(18_000_001)
                    if len(raw)>18_000_000:raise ValueError('oversized_response')
                    if route=='/' and response.status==200:
                        # Keep every iframe request and saved-image URL on this
                        # console's origin, without modifying model observations.
                        page=raw.decode('utf-8').replace("fetch('/camera.jpg')","fetch('/prompted/camera.jpg')")
                        page=page.replace('fetch(path,','fetch(\'/prompted\'+path,')
                        page=page.replace('record.image_url',"('/prompted'+record.image_url)")
                        page=page.replace('读取三个新相机帧，核对标记、类别稳定性','读取一帧新相机图像，核对标记、类别').replace('三帧稳定性','单帧判断')
                        from .prompted_motion_policy import read_motion_options
                        if not read_motion_options()['destination_empty_check_enabled']:
                            page=page.replace('尺寸、空箱与抓取条件','尺寸与抓取条件')
                            page=page.replace('沿用现有运动规划','沿用现有运动规划 · 空箱检查已关闭')
                        raw=page.encode('utf-8')
                    return self.reply(response.status,raw,{'Content-Type':response.headers.get('Content-Type','application/json')})
            except (OSError,ValueError):
                return self.reply(502,dict(error='新模型服务暂不可用，请检查识别服务'))

        def do_GET(self):
            if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
            if self.path=='/camera.jpg':
                try:
                    with opener.open('http://127.0.0.1:8780/frame.jpg',timeout=3) as response:
                        return self.reply(200,response.read(10_000_001),{k:v for k,v in response.headers.items()
                            if k.lower() in ('content-type','x-frame-id','x-frame-age-ms')})
                except OSError:return self.reply(503,dict(error='camera_not_ready'))
            if self.path=='/monitor.js':
                code=Path(__file__).with_name('monitor_assets').joinpath('monitor.js').read_text(encoding='utf-8')
                code=code.replace("label.textContent=names[item.label]||item.label;",
                    "label.textContent=(item.official_label?item.official_label+' · ':'')+(names[item.label]||item.label);")
                code=code.replace('const blockerNames={',
                    "const blockerNames={prompted_calibration_not_current:'参数已变化，需要重新核对照片校准记录',prompted_validation_insufficient:'照片验证不足，当前不能抓取',prompted_model_preview_only:'新模型当前仅支持识别与路线预览，抓放尚待验证',")
                code=code.replace('const blocked=!d.configured||d.blockers?.length,own=d.owner===operationOwner;',
                    """const trialMaterials=d.prompted_trial_materials||[];
const trialAllowed=!d.prompted_trial_materials||(selected==='AUTO'?trialMaterials.length>0:trialMaterials.includes(selected.toLowerCase()));
const heightInput=$('operation-height'),profiles=d.mechanical_heights_mm||{};
if(heightInput.dataset.material!==selected){heightInput.value=profiles[selected==='AUTO'?(trialMaterials[0]||'metal'):selected.toLowerCase()]??'';heightInput.dataset.material=selected;}
heightInput.disabled=d.busy||operationRequest;
const heightValid=heightInput.value!==''&&heightInput.checkValidity();
text('operation-height-note',!heightValid?'请输入 1～120 mm 的有效高度。':'填写值 '+heightInput.value+' mm 将直接用于下一次'+operationMaterials[selected]+'抓取的位置换算；点击开始时固定本次高度，夹爪下降深度沿用当前设置。');
heightInput.oninput=()=>{if(operation)renderOperation(operation)};
const blocked=!d.configured||d.blockers?.length||!trialAllowed||!heightValid,own=d.owner===operationOwner;""")
                code=code.replace("$('operation-preview').disabled=d.busy||operationRequest||!d.configured;",
                    "$('operation-preview').disabled=d.busy||operationRequest||!d.configured||!heightValid;")
                code=code.replace('if(!operation||operationRequest)return;',
                    "if(!operation||operationRequest)return;if(['preview','start'].includes(action)&&!$('operation-height').reportValidity())return;")
                code=code.replace('home_confirmed:action===\'home\'',
                    "home_confirmed:action==='home',...(['preview','start'].includes(action)?{object_height_mm:Number($('operation-height').value)}:{})")
                code=code.replace("d.blockers.map(x=>blockerNames[x]||x)","(d.blockers||[]).map(x=>blockerNames[x]||x)")
                code=code.replace('路径及空箱检查','路径检查')
                code=code.replace("const resetReasons={", "const resetReasons={object_height_must_be_1_to_120_mm:'请输入 1～120 mm 的有效高度。',")
                code=code.replace("$('operation-home').disabled=!d.home_supported||d.busy||operationRequest||operationPending||blocked;",
                    "$('operation-home').disabled=!d.home_supported||d.busy||operationRequest||operationPending||!d.configured||d.blockers?.length;")
                code=code.replace("text('page-message',batch?",
                    "if(d.prompted_trial_labels)text('operation-scope','当前开放 '+d.prompted_trial_labels.length+' 个类别的受监督单件抓取；全局置信度 '+fmt(d.prompted_global_confidence*100,0)+'%；'+(d.prompted_scope_basis==='operator_supervised_trial'?'按操作者要求开放试用，新增类别仍待实物验证。':'沿用已有验证范围。')+'帧龄和观察超时 '+d.prompted_frame_timeout_s+' 秒。'+(d.destination_empty_check_enabled===false?'所有类别的空箱检查已关闭。':'')+(d.full_scene_instance_stability_check_enabled===false?'全画面实例匹配已关闭，待抓目标只需一帧通过；其他轮廓仍参与避让。':'')+'每件完成后核验源区消失与正确箱新增；不自动重试。');if(d.prompted_single_manual_attempt_only){$('operation-arm').disabled=true;$('operation-voice-arm').disabled=true;}text('page-message',batch?")
                code=code.replace("$('upload-image').src='/overlay.jpg?frame='+encodeURIComponent(d.frame_id);",
                    "$('upload-image').src='data:image/jpeg;base64,'+d.overlay_base64;")
                code=code.replace('if(d.prompted_single_manual_attempt_only){',
                    "if(d.projected_region_bounds_check_enabled===false)$('operation-scope').textContent+='区域投影越界拦截已关闭，使用画面内部分。';if(d.prompted_single_manual_attempt_only){")
                return self.reply(200,code.encode(),{'Content-Type':'application/javascript; charset=utf-8'})
            if self.path in ('/health','/live-result','/examples','/benchmark'):
                try:
                    value=health() if self.path=='/health' else observation(recognition('/live-result')) if self.path=='/live-result' else [] if self.path=='/examples' else dict(available=False)
                    return self.reply(200,value)
                except (OSError,ValueError,KeyError):return self.reply(503,dict(error='recognition_not_ready'))
            if self.path=='/':
                if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
                try:
                    info=recognition('/health')
                    page=console_page(info.get('vocabulary_count',24),info.get('trained_on_taco',False))
                except (OSError,ValueError):page=console_page()
                return self.reply(200,page.encode(),{'Content-Type':'text/html; charset=utf-8'})
            if self.path.startswith('/prompted/'):
                return self.proxy_prompted()
            return super().do_GET()

        def do_POST(self):
            if self.path in ('/live','/detect'):
                if not self.allowed():return self.reply(403,dict(error='local_origin_required'))
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=18_000_000 or self.headers.get('Transfer-Encoding'):raise ValueError('invalid_request')
                    payload=json.loads(self.rfile.read(length))
                    if self.path=='/live':return self.reply(200,recognition('/configure',dict(enabled=payload['enabled'])))
                    return self.reply(200,observation(recognition('/detect',payload)))
                except (OSError,ValueError,KeyError):return self.reply(400,dict(error='recognition_request_failed'))
            if self.path.startswith('/prompted/'):
                return self.proxy_prompted()
            return super().do_POST()

    server.RequestHandlerClass=Handler
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8810)
    parser.add_argument('--upstream',default='http://127.0.0.1:8811')
    args=parser.parse_args()
    from .monitor_prompted_operation import prompted_operations
    server=create_console(args.port,upstream=args.upstream,operations=prompted_operations(ROOT))
    print('24-prompt station console ready: http://127.0.0.1:'+str(args.port),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()

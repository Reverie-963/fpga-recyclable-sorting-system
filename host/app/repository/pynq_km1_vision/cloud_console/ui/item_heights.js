/* Per-object measurements belong to a frozen displayed observation. */
(function(root){
  'use strict';
  const clone=x=>JSON.parse(JSON.stringify(x));
  function snapshot(vision){
    if(!vision||!vision.source_stream_id||!vision.model_version||!vision.configuration_sha256||!vision.image_size)
      throw Error('当前画面缺少物品绑定信息，请等待新画面。');
    if(vision.pickup_scope?.status==='unavailable')throw Error('工作区定位中，请等待新画面。');
    const objects=(vision.instances||[]).filter(i=>i.station_grasp_candidate!==false&&
      !String(i.station_zone||'').startsWith('bin_')&&['paper','metal','plastic'].includes(i.label));
    if(!objects.length||objects.length>12)throw Error('请在取物区放好 1–12 件分开的物品。');
    if(objects.some(i=>!i.official_label||!Array.isArray(i.mask_rle)))throw Error('当前画面没有完整轮廓，请等待识别。');
    return {kind:'instance-heights-v1',model_sha256:vision.model_version,
      configuration_sha256:vision.configuration_sha256,source_stream_id:vision.source_stream_id,
      image_size:clone(vision.image_size),items:objects.map((i,n)=>({item_id:n+1,label:i.label,
        official_label:i.official_label,mask_rle:clone(i.mask_rle),object_height_mm:null}))};
  }
  function valid(plan){return !!plan?.items?.length&&plan.items.every(i=>typeof i.object_height_mm==='number'&&
    Number.isFinite(i.object_height_mm)&&i.object_height_mm>=1&&i.object_height_mm<=120);}
  function identity(vision){return [vision?.model_version,vision?.configuration_sha256,vision?.source_stream_id].join(':');}
  function bounds(item,size){let offset=0,x0=size[0],y0=size[1],x1=0,y1=0;
    item.mask_rle.forEach((count,n)=>{if(n%2&&count){for(let p=offset;p<offset+count;p++){
      const x=p%size[0],y=Math.floor(p/size[0]);x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x+1);y1=Math.max(y1,y+1);
    }}offset+=count;});return [x0,y0,x1,y1];}
  class Editor{
    constructor({document,onchange,onerror,saveVoice,clearVoice}){
      this.document=document;this.onchange=onchange;this.onerror=onerror;this.plan=null;this.enabled=false;
      this.saveVoice=saveVoice;this.clearVoice=clearVoice;
      this.panel=document.createElement('section');this.panel.className='item-height-panel';this.panel.hidden=true;
      this.panel.innerHTML='<div class="item-height-heading"><strong>物体高度 <span>mm</span></strong><details class="item-height-help"><summary aria-label="高度填写说明" title="高度填写说明">ⓘ</summary><p>填写桌面到物体顶部的高度，夹爪下降沿用机械配方。编号对应当前画面，当前已登记整批执行后统一核验。</p></details><button type="button" class="text-button" title="刷新当前识别画面的物品登记">登记 ↻</button></div><p class="item-height-hint">桌面 → 物体顶部</p><details class="item-height-picture" hidden><summary>编号参考图</summary><div class="item-height-reference" hidden><img alt="本次登记的物品编号参考图"><svg aria-label="物品编号"></svg></div></details><div class="item-height-rows"></div><p class="item-height-status" role="status"></p>';
      this.button=this.panel.querySelector('button');this.rows=this.panel.querySelector('.item-height-rows');
      this.note=this.panel.querySelector('.item-height-status');
      const shared=document.getElementById('object-height').closest('label');this.shared=shared;
      this.voiceButton=document.createElement('button');this.voiceButton.type='button';this.voiceButton.className='text-button';
      this.voiceButton.textContent='保存高度，供语音启动';this.panel.append(this.voiceButton);
      this.voiceButton.onclick=async()=>{try{await this.saveVoice(this.payload());}catch(e){this.onerror(e.message);}};
      shared.after(this.panel);this.button.onclick=async()=>{try{
        if(this.registration)await this.clearVoice(this.registration.id);
        this.register(this.vision,this.src);
      }catch(e){this.onerror(e.message);}};
    }
    valid(){return valid(this.plan);}
    payload(){if(!this.valid())throw Error('请登记当前画面，并填写每件物品的 1–120 mm 高度。');return clone(this.plan);}
    refresh(state,vision,src,busy,canSave=false){
      this.operationId=state?.operation_id;
      this.enabled=!!state?.per_instance_heights_supported;this.panel.hidden=!this.enabled;this.shared.hidden=this.enabled;
      this.vision=vision;this.src=src;this.button.disabled=busy||!vision;
      const session=state?.voice_height_registration_session,version=state?.voice_height_registration_version??0;
      if(session!==this.registrationSession||version>=(this.registrationVersion??0)){
        this.registration=state?.voice_height_registration;
        this.registrationSession=session;this.registrationVersion=version;
      }
      if(this.voiceButton){
        this.voiceButton.hidden=!state?.voice_height_registration_supported;
        this.voiceButton.disabled=busy||!canSave||!this.valid()||!!this.registration;
        this.voiceButton.textContent=this.registration?'已保存，可语音启动':'保存高度，供语音启动';
      }
      if(this.plan&&vision&&identity(vision)!==identity(this.planVision)){
        this.plan=null;this.rows.replaceChildren();this.note.textContent='模型或相机已变化，请重新登记。';
        this.panel.querySelector('.item-height-reference').hidden=true;this.panel.querySelector('.item-height-picture').hidden=true;
      }
      for(const input of this.rows.querySelectorAll('input,button'))input.disabled=busy||!!this.registration;
      if(state?.active_action==='start'&&state.operation_id!==this.registeredAfterOperation)this.executingOperation=state.operation_id;
      if(this.plan&&!state?.busy&&this.executingOperation&&state?.operation_id===this.executingOperation){
        this.plan=null;this.executingOperation=null;this.rows.replaceChildren();
        this.panel.querySelector('.item-height-reference').hidden=true;this.panel.querySelector('.item-height-picture').hidden=true;
        this.note.textContent='任务结束，请重新登记剩余物品。';
      }
      if(this.plan&&state?.operation_id!==this.registeredAfterOperation&&state?.result?.candidate_order?.length){
        const order=state.result.candidate_order.filter(i=>i.height_item_id).map(i=>
          '#'+i.height_item_id+'（'+i.object_height_mm+' mm）').join(' → ');
        if(order)this.note.textContent='预演顺序：'+order;
      }
      if(this.registration){
        const values=this.registration.items.map(i=>'#'+i.item_id+' '+i.object_height_mm+' mm').join('、');
        this.note.textContent=values+' · 可说“自动分拣”或材质口令，单次有效 · 剩余 '+this.registration.remaining_s+' 秒。修改请重新登记。';
        this.wasVoiceSaved=true;
      }else if(this.wasVoiceSaved){
        this.wasVoiceSaved=false;this.note.textContent='语音登记已使用或失效，请核对当前物品后重新登记。';
        this.plan=null;this.rows.replaceChildren();this.panel.querySelector('.item-height-reference').hidden=true;
        const picture=this.panel.querySelector('.item-height-picture');if(picture)picture.hidden=true;
      }
      if(this.voiceButton)this.voiceButton.disabled=busy||!canSave||!this.valid()||!!this.registration;
    }
    register(vision,src){
      this.registeredAfterOperation=this.operationId;
      this.executingOperation=null;
      this.plan=snapshot(vision);this.planVision=clone(vision);this.rows.replaceChildren();
      this.panel.querySelector('img').src=src;
      const reference=this.panel.querySelector('.item-height-reference');reference.hidden=false;this.panel.querySelector('.item-height-picture').hidden=false;
      this.draw();
      for(const item of this.plan.items){
        const row=this.document.createElement('label');row.className='item-height-row';
        const title=this.document.createElement('span');title.textContent='#'+item.item_id+' '+({metal:'金属',paper:'纸类',plastic:'塑料'}[item.label]||item.official_label);title.title=item.official_label;
        const input=this.document.createElement('input');input.type='number';input.min='1';input.max='120';input.step='any';input.placeholder='—';input.required=true;
        input.setAttribute('aria-label','#'+item.item_id+' 桌面到顶部高度，毫米');
        input.oninput=()=>{item.object_height_mm=input.value===''?null:Number(input.value);this.registeredAfterOperation=this.operationId;const remaining=this.plan.items.filter(i=>!Number.isFinite(i.object_height_mm)||i.object_height_mm<1||i.object_height_mm>120).length;this.note.textContent=remaining?'还需填写 '+remaining+' 件':'高度已填写，可预演';this.onchange();};
        const remove=this.document.createElement('button');remove.type='button';remove.className='text-button';remove.textContent='×';remove.title='移除 #'+item.item_id;remove.setAttribute('aria-label','移除 #'+item.item_id);
        remove.onclick=()=>{this.plan.items=this.plan.items.filter(i=>i!==item);row.remove();this.draw();this.note.textContent=this.plan.items.length?'剩余 '+this.plan.items.length+' 件':'请重新登记画面';this.onchange();};
        row.append(title,input,remove);this.rows.append(row);
      }
      this.note.textContent='还需填写 '+this.plan.items.length+' 件';this.onchange();
    }
    draw(){const svg=this.panel.querySelector('svg');svg.setAttribute('viewBox','0 0 '+this.plan.image_size.join(' '));svg.replaceChildren();
      for(const item of this.plan.items){const [x,y,x1,y1]=bounds(item,this.plan.image_size);
        const rect=this.document.createElementNS('http://www.w3.org/2000/svg','rect');
        for(const [k,v] of Object.entries({x,y,width:x1-x,height:y1-y,fill:'none',stroke:'#ffdf55','stroke-width':2}))rect.setAttribute(k,v);
        const text=this.document.createElementNS('http://www.w3.org/2000/svg','text');text.setAttribute('x',x+4);text.setAttribute('y',y+20);text.setAttribute('fill','#ffdf55');text.setAttribute('font-size',20);text.textContent='#'+item.item_id;svg.append(rect,text);
      }
    }
  }
  root.SortingItemHeights={snapshot,valid,identity,Editor};
  if(typeof module!=='undefined')module.exports=root.SortingItemHeights;
})(typeof window!=='undefined'?window:globalThis);

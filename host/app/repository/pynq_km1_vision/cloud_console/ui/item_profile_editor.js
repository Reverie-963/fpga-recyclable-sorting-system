/* Each current object owns its measurement source; saving is an explicit action. */
(function(root){
  'use strict';
  const H=root.SortingItemHeights,P=root.SortingItemProfiles,copy=P.clone;
  const dimensionValid=i=>Number.isFinite(i.object_height_mm)&&i.object_height_mm>=1&&i.object_height_mm<=120&&
    (!('grasp_width_mm' in i)||(Number.isFinite(i.grasp_width_mm)&&i.grasp_width_mm>=1&&i.grasp_width_mm<=150));
  function overlap(a,b){let i=0,j=0,ar=a[0],br=b[0],intersection=0,union=0;
    while(i<a.length&&j<b.length){const n=Math.min(ar,br);if(i%2&&j%2)intersection+=n;if(i%2||j%2)union+=n;
      ar-=n;br-=n;if(!ar)ar=a[++i];if(!br)br=b[++j];}return union?intersection/union:0;}
  class Editor extends H.Editor{
    constructor(options){
      super({...options,onchange:()=>{if(!this.internal)this.changed();}});
      this.notify=options.onchange;this.image=options.image;this.library=new P.Client(options.profileRequest);
      this.profileLoaded=false;this.autoKey='';this.signature='';this.manualItems=new Set();this.overrides=new Set();
      this.panel.querySelector('strong').firstChild.textContent='物品尺寸';this.panel.querySelector('strong span').hidden=true;
      this.panel.querySelector('.item-height-hint').hidden=true;
      this.panel.querySelector('.item-height-help p').textContent='优先匹配已有尺寸，未匹配时默认选择同名物品或同材质的第一项，可随时更换或临时填写。调整只用于本次，点保存才更新尺寸库。高度从桌面量到顶部，宽度量当前抓点处；不同规格或姿态分别保存。';
      this.libraryView=this.node('details',null,'measurement-library');this.libraryTitle=this.node('summary','尺寸库');
      this.libraryList=this.node('div',null,'measurement-library-list');this.libraryView.append(this.libraryTitle,this.libraryList);this.button.before(this.libraryView);
      this.button.setAttribute('aria-label','重新识别物品');this.button.onclick=()=>this.rescan();
      const menu=this.document.getElementById('more-menu');if(menu){this.voiceState=this.node('p','语音未连接','subtle');menu.append(this.voiceState,this.voiceButton);}
      this.library.load().then(()=>{this.profileLoaded=true;this.renderLibrary();this.autoKey='';this.notify();}).catch(e=>{this.note.textContent=e.message+'，可临时填写。';});
    }
    node(tag,text,cls){const e=this.document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;}
    changed(){const ready=this.document.getElementById('scene-ready');if(ready)ready.checked=false;this.notify?.();}
    item(id){const item=this.plan?.items.find(i=>i.item_id===id);if(!item)throw Error('物品已变化，请重新识别。');return item;}
    profile(id){return this.library.data.profiles.find(p=>p.id===this.selected?.get(id));}
    updateDraft(){this.draft=!!this.plan?.items.some(i=>this.manualItems.has(i.item_id)||this.overrides.has(i.item_id)||!this.selected.has(i.item_id));}
    summary(item){return '高 '+(item.object_height_mm??'—')+('grasp_width_mm' in item?' · 宽 '+item.grasp_width_mm:'')+' mm';}
    renderLibrary(){this.libraryTitle.textContent='尺寸库';this.libraryTitle.title=this.library.data.profiles.length+' 件已保存物品';this.libraryList.replaceChildren();
      if(!this.library.data.profiles.length)this.libraryList.append(this.node('p','暂无保存的尺寸'));
      for(const p of this.library.data.profiles){const row=this.node('div');row.append(this.node('strong',p.name+' · '+p.pose),this.node('p',this.summary({object_height_mm:p.height_mm,...(p.width_mm==null?{}:{grasp_width_mm:p.width_mm})})+(p.features?'':' · 手动选择')));this.libraryList.append(row);}}
    clearPlan(message){this.plan=null;this.signature='';this.rows.replaceChildren();this.note.textContent=message;
      this.panel.querySelector('.item-height-reference').hidden=true;this.panel.querySelector('.item-height-picture').hidden=true;}
    async rescan(){try{if(this.registration)await this.clearVoice(this.registration.id);this.plan=null;this.register(this.vision,this.src);this.changed();}
      catch(e){this.clearPlan(e.message);this.changed();this.onerror(e.message);}}
    valid(){return super.valid()&&this.plan.items.every(i=>dimensionValid(i)&&(!('grasp_width_mm' in i)||this.widthSupported));}
    payload(){if(!this.valid())throw Error('请补全每件物品的尺寸；实测宽度需工位支持。');return copy(this.plan);}
    picture(vision){const image=this.image?.(),[w,h]=vision.image_size;
      if(!image?.complete||image.naturalWidth!==w||image.naturalHeight!==h)throw Error('等待当前识别画面加载');
      const canvas=this.document.createElement('canvas');canvas.width=w;canvas.height=h;
      const context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(image,0,0);return {pixels:context.getImageData(0,0,w,h).data};}
    descriptor(item,vision,picture){return P.features(picture.pixels,item.mask_rle,vision.image_size);}
    apply(item,profile){item.object_height_mm=profile.height_mm;
      if(profile.width_mm!=null)item.grasp_width_mm=profile.width_mm;else delete item.grasp_width_mm;
      if(this.widthSupported)item.measurement_profile={id:profile.id,revision:profile.revision,name:profile.name,pose:profile.pose};else delete item.measurement_profile;
      this.selected.set(item.item_id,profile.id);this.manualItems.delete(item.item_id);this.overrides.delete(item.item_id);}
    selectSource(id,value){const item=this.item(id),profile=this.library.data.profiles.find(p=>p.id===value&&p.material===item.label);
      if(value&&value!=='manual'&&!profile)throw Error('请选择相同材质的尺寸档案。');
      this.selected.delete(id);this.manualItems.delete(id);this.overrides.delete(id);item.object_height_mm=null;delete item.grasp_width_mm;delete item.measurement_profile;
      if(profile)this.apply(item,profile);else if(value==='manual')this.manualItems.add(id);
      this.updateDraft();this.registeredAfterOperation=this.operationId;this.note.textContent='';this.signature='';this.autoKey='';this.renderRows();this.changed();}
    editMeasurement(id,height,width){const item=this.item(id);item.object_height_mm=height;
      if(width==null)delete item.grasp_width_mm;else item.grasp_width_mm=width;
      delete item.measurement_profile;this.overrides.add(id);this.updateDraft();this.registeredAfterOperation=this.operationId;this.note.textContent='';this.changed();}
    async persist(id,name,pose,existingId){const item=this.item(id),existing=this.profile(id),features=this.sourceFeatures.get(id)||existing?.features;
      if(existingId&&existingId!==existing?.id)throw Error('尺寸档案已变化，请重新选择。');
      if(!features)throw Error('参考画面尚未就绪，请重新识别。');if(!dimensionValid(item))throw Error('请填写有效的高度和宽度。');
      const value={name:name.trim(),pose:pose.trim(),material:item.label,height_mm:item.object_height_mm,width_mm:item.grasp_width_mm??null,
        official_label:item.official_label,reference_model:this.sourceFeatures.get(id)?this.plan.model_sha256:existing.reference_model,features:copy(features),thumbnail:''};
      if(!this.profileLoaded)await this.library.load();this.profileLoaded=true;
      const saved=await this.library.save(value,existingId);
      this.renderLibrary();if(this.plan?.items.includes(item)){this.apply(item,saved);this.updateDraft();this.autoKey='';this.signature='';this.renderRows();this.changed();}return saved;}
    register(vision,src){this.internal=true;
      try{const previous=this.plan,oldVision=this.planVision,oldSelected=this.selected||new Map();if(!previous)this.signature='';
        this.plan=H.snapshot(vision);this.planVision=copy(vision);this.selected=new Map();this.manualItems=new Set();this.overrides=new Set();this.sourceFeatures=new Map();this.draft=false;
        this.registeredAfterOperation=this.operationId;this.executingOperation=null;
        this.panel.querySelector('img').src=src;this.panel.querySelector('.item-height-reference').hidden=false;this.panel.querySelector('.item-height-picture').hidden=this.plan.items.length<2;
        let picture;try{picture=this.picture(vision);}catch{}
        const used=new Set();
        for(const item of this.plan.items){let descriptor=null;if(picture)try{descriptor=this.descriptor(item,vision,picture);}catch{}
          this.sourceFeatures.set(item.item_id,descriptor);if(!this.profileLoaded)continue;
          const candidates=previous&&H.identity(oldVision)===H.identity(vision)?previous.items.filter(p=>p.label===item.label&&overlap(p.mask_rle,item.mask_rle)>=.85):[];
          const retained=candidates.length===1&&!used.has(candidates[0].item_id)?this.library.data.profiles.find(p=>p.id===oldSelected.get(candidates[0].item_id)&&p.material===item.label):null;
          const profiles=this.library.data.profiles.filter(p=>p.material===item.label);
          const profile=retained||P.match(profiles,descriptor,item.label,vision.model_version)||profiles.find(p=>p.official_label===item.official_label)||profiles[0];
          if(retained)used.add(candidates[0].item_id);if(profile)this.apply(item,profile);
        }
        this.draw();const signature=JSON.stringify([this.plan.items.map(i=>[i.item_id,i.label,i.object_height_mm,i.grasp_width_mm,this.selected.get(i.item_id)]),this.library.data.revision]);
        if(signature!==this.signature){const was=this.signature;this.signature=signature;this.renderRows();if(was)this.document.getElementById('scene-ready').checked=false;}
        this.note.textContent=this.valid()?'':'请选择物品尺寸';
      }finally{this.internal=false;}}
    refresh(state,vision,src,busy,canSave=false){this.widthSupported=!!state?.per_instance_widths_supported;
      super.refresh(state,vision,src,busy,canSave);this.busy=busy;
      if(this.voiceState)this.voiceState.textContent=state?.voice_service?.connected?'语音已连接 · 自动匹配尺寸库':'语音未连接';
      this.button.textContent='↻';this.button.title=this.registration?'取消语音登记并重新识别':'重新识别当前物品；临时修改会清空';
      if(this.voiceButton){this.voiceButton.textContent=this.registration?'语音尺寸已保存':'保存尺寸供语音使用';this.voiceButton.title='保存当前尺寸，5分钟内可用语音启动，仅一次';}
      if(this.registration){this.note.title=this.note.textContent;this.note.textContent='语音已启用 · '+this.registration.remaining_s+' s';}
      const editing=!!this.document.activeElement&&!!this.panel.contains?.(this.document.activeElement);
      if(vision&&!busy&&!this.registration&&(!this.draft||!this.plan)&&(!this.plan||!editing)){
        const key=(vision.frame_key||src)+':'+this.library.data.revision+':'+this.widthSupported;
        if(key!==this.autoKey||!this.plan){this.autoKey=key;try{this.register(vision,src);}catch(e){const empty=(vision.instances||[]).every(i=>i.station_grasp_candidate===false||String(i.station_zone||'').startsWith('bin_'));
          this.clearPlan(vision.pickup_scope?.status==='unavailable'?e.message:empty?'等待取物区物品':e.message);}}
      }
      for(const e of this.rows.querySelectorAll('input,button,select'))e.disabled=busy||!!this.registration;
      if(this.plan?.items.some(i=>'grasp_width_mm' in i)&&!this.widthSupported)this.note.textContent='工位尚不支持实测宽度，请等待服务更新。';
      if(this.voiceButton)this.voiceButton.disabled=busy||!canSave||!this.valid()||!!this.registration;
    }
    renderRows(){this.rows.replaceChildren();
      for(const item of this.plan.items){const id=item.item_id,profile=this.profile(id),manual=this.manualItems.has(id),row=this.node('div',null,'measurement-row');
        const heading=this.node('div',null,'measurement-row-heading');if(this.plan.items.length>1)heading.append(this.node('span','#'+id));
        const select=this.node('select');select.setAttribute('aria-label','#'+id+' 尺寸来源');
        for(const [value,title] of [['','选择物品尺寸'],...this.library.data.profiles.filter(p=>p.material===item.label).map(p=>[p.id,p.name+' · '+p.pose]),['manual','临时填写']]){
          const option=this.node('option',title);option.value=value;select.append(option);}
        select.value=manual?'manual':this.selected.get(id)||'';select.onchange=()=>{try{this.selectSource(id,select.value);}catch(e){this.onerror(e.message);}};heading.append(select);row.append(heading);
        const summary=this.node('p',profile?this.summary(item):manual?item.official_label:'','measurement-summary');row.append(summary);
        const details=this.node('details',null,'measurement-details');details.hidden=!profile&&!manual;details.open=manual;details.append(this.node('summary',manual?'本次尺寸':'调整尺寸'));
        const fields=this.node('div',null,'measurement-fields');
        const input=(text,value,type='text',max,target=fields)=>{const label=this.node('label',text),control=this.node('input');control.type=type;control.value=value??'';
          if(type==='number'){control.min=1;control.max=max;control.step='any';control.placeholder='—';}else control.maxLength=text==='姿态'?24:60;
          control.setAttribute('aria-label','#'+id+' '+text);label.append(control);target.append(label);return control;};
        const height=input('高度 mm',item.object_height_mm,'number',120),width=input('实测宽度 mm',item.grasp_width_mm,'number',150);
        width.placeholder='选填';width.title='当前抓点处的实测夹持宽度，细管可填外径';
        const localState=this.node('div',null,'measurement-local-state'),sourceNote=this.node('span',this.overrides.has(id)?'仅用于本次':'','measurement-help');
        const restore=this.node('button','恢复已存尺寸','text-button');restore.type='button';restore.hidden=!profile||!this.overrides.has(id);restore.onclick=()=>this.selectSource(id,profile.id);localState.append(sourceNote,restore);
        height.oninput=width.oninput=()=>{this.editMeasurement(id,height.value===''?null:Number(height.value),width.value===''?null:Number(width.value));
          summary.textContent=this.summary(this.item(id));sourceNote.textContent='仅用于本次';restore.hidden=!profile;};details.append(fields,localState);
        const archive=this.node('details',null,'measurement-archive');archive.append(this.node('summary',profile?'保存修改':'保存到尺寸库'));
        const archiveFields=this.node('div',null,'measurement-fields'),name=input('名称',profile?.name||item.official_label,'text',undefined,archiveFields),pose=input('姿态',profile?.pose||'平放','text',undefined,archiveFields);archive.append(archiveFields);
        const actions=this.node('div',null,'measurement-actions');
        const save=(title,existingId)=>{const b=this.node('button',title,'text-button');b.type='button';b.onclick=async()=>{try{await this.persist(id,name.value,pose.value,existingId);}catch(e){this.onerror(e.message);}};actions.append(b);};
        if(profile)save('更新原档案',profile.id);save(profile?'另存为新物品':'保存');
        if(profile){const remove=this.node('button','移除档案','text-button');remove.type='button';remove.onclick=async()=>{try{await this.library.remove(profile.id);this.renderLibrary();this.clearPlan('档案已移除');this.autoKey='';this.changed();}catch(e){this.onerror(e.message);}};actions.append(remove);}
        archive.append(actions);details.append(archive);row.append(details);this.rows.append(row);
      }
    }
  }
  root.SortingMeasurementEditor=Editor;
  if(typeof module!=='undefined')module.exports={Editor,overlap};
})(typeof window!=='undefined'?window:globalThis);

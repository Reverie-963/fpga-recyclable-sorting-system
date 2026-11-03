(() => {
  const $=id=>document.getElementById(id),labels={AUTO:'全部分类',PAPER:'纸类',PLASTIC:'塑料',METAL:'金属',paper:'纸类',plastic:'塑料',metal:'金属'};
  const owner='cloud-'+crypto.randomUUID();
  let state=null,online=false,material='AUTO',frameKind='overlay',frameKey='',requestBusy=false,heartbeatBusy=false,activeRequest=null;
  let lastPollAt=0,baseAge=Infinity,modelKey='',modelFamily='yoloe',recordsKey='',toastTimer,view='console';
  const local=window.__LOCAL_CONSOLE__||null;
  let resetBusy=false,recovery=null,resetPhase='',resetStopSent=false,taskStartedAt=0,taskOperationId='',taskEndedAt=null,taskFinishedElapsed=null;
  let loadedVision=null,pendingFrame=null,taskSubmitted=null,progressAt=0;
  let reviewContext=null,reviewSubmitting=false;
  const clock=value=>value?new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(value)): '—';
  const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e;};
  function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,5000);}
  const itemHeightEditor=window.SortingItemHeights?new (window.SortingMeasurementEditor||window.SortingItemHeights.Editor)({document,onchange:()=>buttons(),onerror:toast,
    profileRequest:api,image:()=>$('live-frame'),
    saveVoice:plan=>voiceHeightRegistration('register_heights',plan),
    clearVoice:id=>voiceHeightRegistration('clear_heights',null,id)}):null;
  async function voiceHeightRegistration(action,plan,id){
    if(action==='register_heights'&&!$('scene-ready').checked)throw Error('请先核对并确认现场');
    const payload={action,owner,ready:$('scene-ready').checked,request_id:crypto.randomUUID(),
      expected_operation_id:state?.operation?.operation_id??null,
      height_registration_id:id??state?.operation?.voice_height_registration?.id??null};
    if(plan)payload.height_plan=plan;
    const reply=await run('operation',payload);
    if(state)state.operation=reply;
    // Refresh the editor before allowing a new frame registration after revoke.
    buttons();
    $('command-message').textContent=action==='register_heights'?'高度已保存，5分钟内可说“自动分拣”或材质口令；'+(reply?.batch_mode?'一次完成当前已登记的一批，最后统一核验。':'每次执行一件。'):'语音高度登记已取消，可重新填写。';
    poll();return reply;
  }
  async function api(path,options={}){
    if(!local&&path==='/api/operation')return run('operation_state',{}, {quiet:true});
    if(!local&&path==='/api/item-profiles')return run('profiles',options.body?JSON.parse(options.body):{action:'read'}, {quiet:true});
    let r;try{r=await fetch(path,{cache:'no-store',...options,headers:{...options.headers,...(local?{'X-Console-Session':local.token}:{})}});}catch{
      throw Error(options.method==='POST'?'中控连接中断，本次响应未确认；请先核对任务状态，不要重复提交。':'中控连接中断，请刷新页面或检查服务。');
    }let d;try{d=await r.json();}catch{throw Error(options.method==='POST'?'中控响应未确认；请先核对任务状态，不要重复提交。':'连接暂不可用，请稍后重试');}if(!r.ok)throw Error(d.error||'操作未完成');return d;
  }
  const fresh=()=>online&&performance.now()-lastPollAt+baseAge<6500;
  function setView(next){view=next;$('console-view').hidden=next!=='console';$('history-view').hidden=next!=='history';for(const b of document.querySelectorAll('[data-view]'))b.classList.toggle('active',b.dataset.view===next);sidebarOpen=false;updateSidebar();}
  for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>setView(b.dataset.view);
  const sidebarMedia=window.matchMedia?.('(max-width: 760px)');
  let sidebarCollapsed=false,sidebarOpen=false;
  function updateSidebar(){
    const mobile=sidebarMedia?.matches===true,expanded=mobile?sidebarOpen:!sidebarCollapsed;
    $('app-shell').dataset.sidebarCollapsed=String(!mobile&&sidebarCollapsed);
    $('sidebar').classList.toggle('shown',mobile&&sidebarOpen);
    $('sidebar').inert=!expanded;
    $('sidebar').setAttribute('aria-hidden',String(!expanded));
    $('sidebar-toggle').setAttribute('aria-expanded',String(expanded));
    $('sidebar-toggle').setAttribute('aria-label',expanded?'收起侧栏':'展开侧栏');
    $('sidebar-toggle').setAttribute('title',expanded?'收起侧栏':'展开侧栏');
  }
  $('sidebar-toggle').onclick=()=>{
    if(sidebarMedia?.matches===true)sidebarOpen=!sidebarOpen;else sidebarCollapsed=!sidebarCollapsed;
    updateSidebar();
  };
  sidebarMedia?.addEventListener?.('change',()=>{sidebarOpen=false;updateSidebar();});
  updateSidebar();
  for(const b of document.querySelectorAll('[data-close]'))b.onclick=()=>$(b.dataset.close).close();
  const settings=()=>$('settings-dialog').showModal();$('settings-open').onclick=settings;
  $('connect-help').onclick=settings;
  const resetDialog=$('station-reset-dialog');
  const resetClose=()=>{if(!resetBusy||['confirmation','position','failed'].includes(resetPhase)){if(recovery)recovery.cancelled=true;resetBusy=false;activeRequest=null;resetDialog.close();recovery=null;buttons();}};
  $('station-reset-close').onclick=resetClose;
  resetDialog.oncancel=e=>{e.preventDefault();resetClose();};
  // Reset-step presentation follows controller progress; it never sends commands.
  let resetAnimationStep=0,resetAnimationRun=0;
  function resetProgress(phase,message){
    const stages={reading:[0,'正在读取状态'],stopping:[0,'正在停止任务'],confirmation:[0,'等待现场确认'],unlocking:[1,'正在归档并解锁'],homing:[2,'正在回初始'],finishing:[3,'正在恢复空闲'],position:[3,'等待位置核对']};
    if(phase==='reading'){resetAnimationStep=0;resetAnimationRun++;}
    if(stages[phase])resetAnimationStep=stages[phase][0];
    const names=['停止与现场检查','归档并解锁','回初始位置','恢复空闲状态'];
    for(let i=0;i<names.length;i++){
      const done=phase==='complete'||i<resetAnimationStep;
      const current=i===resetAnimationStep&&phase!=='complete';
      const status=done?'done':current?(phase==='failed'?'failed':'active'):'pending';
      const caption=done?'已完成':current?(phase==='failed'?'未完成':stages[phase]?.[1]||'正在处理'):'未开始';
      $('reset-step-'+i).dataset.state=status;
      $('reset-step-'+i).setAttribute('aria-current',current?'step':'false');
      $('reset-step-'+i).setAttribute('aria-label',names[i]+'：'+caption);
      $('reset-step-icon-'+i).textContent=done?'✓':status==='failed'?'!':status==='active'?'':String(i+1);
      $('reset-step-status-'+i).textContent=caption;
    }
    resetPhase=phase;$('station-reset-progress').textContent=message;resetDialog.dataset.phase=phase;
  }
  // End reset-step presentation.

  async function recoverySend(action,payload){
    if(action==='stop')resetStopSent=true;
    if(action==='home'){activeRequest=payload.request_id;resetStopSent=false;}
    return run('operation',{action,...payload},{quiet:action==='stop'});
  }
  async function resetFailure(error){
    if(recovery)recovery.cancelled=true;
    if(resetPhase==='homing'&&!resetStopSent){
      let current;try{current=await api('/api/operation');}catch{}
      if(window.StationResetFlow.failureNeedsStop(current,owner,recovery?.expected)){
        try{await recoverySend('stop',{owner});}catch{}
      }
    }
    activeRequest=null;resetBusy=false;$('scene-ready').checked=false;
    $('station-reset-confirm-group').hidden=true;$('station-reset-run').hidden=true;$('station-reset-finish').hidden=true;$('station-reset-abort').hidden=true;
    $('station-reset-close').disabled=false;resetProgress('failed',window.StationResetFlow.describe(error.message)+' 未自动重试；历史记录已保留。');buttons();poll();
  }
  $('station-reset-open').onclick=async()=>{
    if(resetBusy||!fresh()||!state?.operation)return;
    resetBusy=true;activeRequest=null;resetStopSent=false;$('scene-ready').checked=false;
    $('station-reset-confirmed').checked=false;$('station-reset-run').disabled=true;
    for(const id of ['station-reset-confirm-group','station-reset-run','station-reset-finish'])$(id).hidden=true;
    $('station-reset-abort').hidden=false;$('station-reset-close').disabled=true;
    resetProgress('reading','正在读取工位状态…');resetDialog.showModal();buttons();
    recovery=new window.StationResetFlow.Recovery({read:()=>api('/api/operation'),send:recoverySend,owner,id:()=>crypto.randomUUID(),progress:resetProgress});
    try{
      await recovery.prepare(state.operation.operation_id??null);
      $('station-reset-confirm-group').hidden=false;$('station-reset-run').hidden=false;$('station-reset-abort').hidden=true;$('station-reset-close').disabled=false;
    }catch(error){await resetFailure(error);}
  };
  $('station-reset-confirmed').onchange=()=>{$('station-reset-run').disabled=!$('station-reset-confirmed').checked;};
  $('station-reset-run').onclick=async()=>{
    if(!recovery||resetPhase!=='confirmation'||!$('station-reset-confirmed').checked)return;
    $('station-reset-run').hidden=true;$('station-reset-confirm-group').hidden=true;$('station-reset-close').disabled=true;$('station-reset-abort').hidden=false;
    try{
      await recovery.run(true);
      $('station-reset-abort').hidden=true;$('station-reset-finish').hidden=false;$('station-reset-close').disabled=false;
    }catch(error){await resetFailure(error);}
  };
  $('station-reset-finish').onclick=async()=>{
    if(!recovery||resetPhase!=='position')return;
    $('station-reset-finish').disabled=true;
    try{
      await recovery.verifyReady();
      resetBusy=false;activeRequest=null;recovery=null;$('scene-ready').checked=true;
      resetProgress('complete','重置完成，工位已就绪，可以准备下一次任务。');
      $('station-reset-finish').hidden=true;
      const completedAnimationRun=resetAnimationRun;
      setTimeout(()=>{if(resetPhase==='complete'&&resetAnimationRun===completedAnimationRun)resetDialog.close();},900);
      const height=Number($('object-height').value);$('command-message').textContent=height>=1&&height<=120?'工位已重置并核对，可按当前类别和高度开始下一次任务。':'工位已重置并核对，填写本次物体高度后即可开始。';buttons();poll();
    }catch(error){await resetFailure(error);}
    finally{$('station-reset-finish').disabled=false;}
  };
  $('station-reset-abort').onclick=async()=>{
    if(!recovery)return;recovery.cancelled=true;
    try{if(!resetStopSent)await recoverySend('stop',{owner});await resetFailure(Error('已请求停止重置，请核对工位是否停稳。'));}
    catch{await resetFailure(Error('停止回应未确认，请在现场核对工位。'));}
  };
  $('station-reset-open').hidden=!window.StationResetFlow;
  if(local){$('advanced-settings').hidden=false;$('advanced-link').href=local.station_url||'http://127.0.0.1:8810';document.title='Reverie · 本地分拣控制台';$('connect-help').textContent='查看工位状态 ↗';document.querySelector('.help-note').textContent='此页面直接连接本机工位服务。模型切换与动作沿用当前工位检查，任务记录保存在原工位。';}
  if(!local){
    const button=node('button','读取连接诊断','text-button'),details=node('pre');
    details.hidden=true;details.style.cssText='white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:50vh;overflow:auto';
    const note=node('p','保持页面在前台。锁屏或断网后，原工位会在控制租约到期时停止继续执行。','help-note');
    $('settings-dialog').append(button,details,note);
    button.onclick=async()=>{button.disabled=true;details.hidden=false;details.textContent='正在读取…';try{const data=await api('/api/diagnostics');details.textContent=JSON.stringify(data,null,2);}catch(e){details.textContent=e.message;}finally{button.disabled=false;}};
  }
  $('more-trigger').onclick=()=>{const hidden=!$('more-menu').hidden;$('more-menu').hidden=hidden;$('more-trigger').setAttribute('aria-expanded',String(!hidden));};
  let modelDraft=null,modelDraftRevision=null,modelApplying=false;
  const modelProfiles={
    yoloe26:['微调版','26类素材微调，适合与原始提示配置对比。'],
    yoloe24:['原始提示版','保留原始提示配置，适合观察提示识别效果。'],
    yoloe26_tuned:['阈值调整版','公开素材续训并调整筛选阈值，便于观察漏检与误检。'],
    yoloe26_reference:['参考图辅助版','结合参考图库辅助材质判断，适合固定样件对照。'],
    v8_3:['三类专项版','直接区分罐、塑料和纸板，适合三类工位对照。'],
    '11s_3':['三类专项版','直接区分罐、塑料和纸板，适合三类工位对照。'],
    v8_26:['公开数据版','公开26类训练候选，适合更细类别的识别对比。'],
    '11s_26':['公开数据版','公开26类训练候选，适合更细类别的识别对比。']
  };
  // Historical records, not live measurements. Each entry identifies its own
  // evaluation scope; never copy scores to a different weight/configuration.
  const modelEvidence={
    yoloe26:{values:[['分割 mAP50–95','12.61%'],['三材质 F1','31.64%'],['推理中位','69.2 ms']],scope:'10/31历史评测 · 公开389图 / 工位121图',detail:'权重 acb01cf3；公开 mask mAP50–95；工位 macro F1（conf 0.30、mask IoU≥0.5）；同机 GPU FP16 predict 调用中位数，含调用内预后处理，不含读图、相机或机械。',source:'docs/team/reports/2026-10-31-public26-results.json'},
    v8_26:{values:[['分割 mAP50–95','5.28%'],['三材质 F1','17.56%'],['推理中位','31.7 ms']],scope:'10/31历史评测 · 公开389图 / 工位121图',detail:'权重 21de18e6；公开 mask mAP50–95；工位 macro F1（conf 0.30、mask IoU≥0.5）；同机 GPU FP16 predict 调用中位数，含调用内预后处理，不含读图、相机或机械。',source:'docs/team/reports/2026-10-31-public26-results.json'},
    '11s_26':{values:[['分割 mAP50–95','5.33%'],['三材质 F1','19.57%'],['推理中位','32.0 ms']],scope:'10/31历史评测 · 公开389图 / 工位121图',detail:'权重 7368e131；公开 mask mAP50–95；工位 macro F1（conf 0.30、mask IoU≥0.5）；同机 GPU FP16 predict 调用中位数，含调用内预后处理，不含读图、相机或机械。',source:'docs/team/reports/2026-10-31-public26-results.json'},
    v8_3:{values:[['三材质 F1','86.76%'],['验证图像','121 张'],['评测阈值','0.30']],scope:'10/29历史评测 · 三类专项验证',detail:'权重 9288e697；macro F1 按固定阈值报告的三类 TP/FP/FN 复算，mask IoU≥0.5。与公开26类训练配方不同，不作为架构排名。',source:'local-only/recyclables-model-comparison/20261029-104817/models/yolov8s-seg/result.json'},
    '11s_3':{values:[['三材质 F1','77.25%'],['验证图像','121 张'],['评测阈值','0.30']],scope:'10/29历史评测 · 三类专项验证',detail:'权重 d7436017；macro F1 按固定阈值报告的三类 TP/FP/FN 复算，mask IoU≥0.5。与公开26类训练配方不同，不作为架构排名。',source:'local-only/recyclables-model-comparison/20261029-104817/models/yolo11s-seg/result.json'},
    yoloe26_tuned:{values:[['回放合格','82 / 128'],['单件选择 F1','68.05%'],['评测阈值','0.10']],scope:'10/31历史回放 · 128张既有姿态图',detail:'权重 9d5734b3；合格要求源区最高分材质正确且 mask IoU≥0.5。CPU FP32，无参考辅助；图像已重复使用，非独立泛化测试。',source:'docs/team/reviews/2026-10-31-yoloe-public16-training.md'},
    yoloe26_reference:{values:[['登记参考图','64 张'],['训练图像','2490 张'],['公开验证','128 张']],scope:'10/31登记 / 训练记录 · 辅助版精度未单独列入',detail:'基础权重 9d5734b3；2490/128为原始权重训练/验证规模，含增强视图，不是独立实物数；64为登记参考图库，不沿用无参考辅助版的F1。当前运行阈值/图库数量以卡片上方服务回报为准。',source:'docs/team/reports/2026-10-31-yoloe-public16-results.json；tools/prepare_console_models.py'},
    yoloe24:{values:[['提示类别','24 类'],['独立评测','待补']],scope:'原始24类配置 · 尚无可对应的评分记录',detail:'不套用26类初始化或微调模型的评测。',source:'tools/prepare_console_models.py'}
  };
  const familyId=m=>m.id.startsWith('yoloe')?'yoloe':m.id.startsWith('v8_')?'v8':m.id.startsWith('11s_')?'11s':'other';
  const familyNames={yoloe:'YOLOE-11m',v8:'YOLOv8s','11s':'YOLO11s',other:'其他型号'};
  const technicalFamilies={yoloe:'YOLOE-11m',v8:'YOLOv8s','11s':'YOLO11s'};
  // Display aliases only: model IDs, revisions and execution bindings stay unchanged.
  const modelVersions={yoloe24:'Base',yoloe26:'Plus',yoloe26_tuned:'Pro',yoloe26_reference:'Pro Max',v8_3:'Pro',v8_26:'Plus','11s_3':'Pro','11s_26':'Plus'};
  const modelVersionRank={'Pro Max':4,Pro:3,Plus:2,Base:1};
  const modelLabel=m=>modelVersions[m.id]?familyNames[familyId(m)]+' · '+modelVersions[m.id]:(m.name||m.id).replace(/ · ROCm GPU$/,'');
  function openModelDetails(id){
    const model=state?.models?.models?.find(m=>m.id===id);if(!model)return;
    $('model-details-title').textContent=modelLabel(model);
    $('model-details-type').textContent=technicalFamilies[familyId(model)]||'识别模型';
    const content=$('model-details-content'),facts=node('dl',null,'model-detail-facts');
    const rows=[['识别类别',Number.isInteger(model.class_count)?model.class_count+' 类':'未提供'],['配置',modelProfiles[model.id]?.[0]||model.name||model.id],['原始名称',model.name||model.id],['配置 ID',model.id]];
    if(model.id===state.models.selected){const config=state.models.selected_configuration;
      if(typeof config?.confidence==='number'&&Number.isFinite(config.confidence))rows.push(['当前置信阈值',config.confidence.toFixed(2)]);
      if(Number.isInteger(config?.reference_count)&&config.reference_count>0)rows.push(['当前参考图库',config.reference_count+' 张']);
    }
    for(const [label,value] of rows){const row=node('div');row.append(node('dt',label),node('dd',value));facts.append(row);}
    content.replaceChildren(facts);
    if(modelProfiles[model.id])content.append(node('p',modelProfiles[model.id][1],'model-detail-description'));
    if(model.disabled_reason)content.append(node('p',model.disabled_reason,'model-detail-description'));
    const evidence=modelEvidence[model.id];
    if(evidence){
      const metrics=node('dl',null,'model-detail-metrics');
      for(const [label,value] of evidence.values){const row=node('div');row.append(node('dt',label),node('dd',value));metrics.append(row);}
      content.append(node('h3','历史评测与记录'),node('p',evidence.scope,'model-detail-scope'),metrics,node('p',evidence.detail,'model-detail-description'),node('p','来源：'+evidence.source,'model-detail-source'));
    }else content.append(node('p','暂无对应的独立评测记录。','model-detail-description'));
    $('model-details-dialog').showModal();
  }
  $('model-current-details').onclick=()=>openModelDetails(state?.models?.selected);
  $('model-selection-details').onclick=()=>openModelDetails(modelDraft);
  function closeMenus(){$('model-menu').hidden=true;$('model-trigger').setAttribute('aria-expanded','false');$('more-menu').hidden=true;}
  function openModels(){
    if($('model-trigger').disabled)return;
    modelDraft=state?.models?.selected;modelDraftRevision=state?.models?.revision;
    const current=(state?.models?.models||[]).find(m=>m.id===modelDraft);
    if(current)showConfigurations(familyId(current));
    $('model-menu').hidden=false;$('model-trigger').setAttribute('aria-expanded','true');updateModelSelection();
  }
  document.addEventListener('pointerdown',e=>{if($('model-details-dialog').open)return;if(!e.target.closest('.model-picker')){$('model-menu').hidden=true;$('model-trigger').setAttribute('aria-expanded','false');}if(!e.target.closest('.topbar-right'))$('more-menu').hidden=true;});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('model-details-dialog').open){const wasOpen=!$('model-menu').hidden;closeMenus();if(wasOpen)$('model-trigger').focus();}});
  $('model-trigger').onclick=()=>{$('model-menu').hidden?openModels():closeMenus();};
  $('model-trigger').onkeydown=e=>{if(e.key==='ArrowDown'){e.preventDefault();openModels();$('model-families').querySelector('button')?.focus();}};
  $('model-menu').onkeydown=e=>{const buttons=[...$('model-menu').querySelectorAll('button:not(:disabled)')],i=buttons.indexOf(document.activeElement);if(e.key==='Escape'){closeMenus();$('model-trigger').focus();}else if(['ArrowDown','ArrowUp','ArrowRight','ArrowLeft'].includes(e.key)){e.preventDefault();if(e.key==='ArrowRight')$('model-configs').querySelector('button:not(:disabled)')?.focus();else if(e.key==='ArrowLeft')$('model-families').querySelector('button.active')?.focus();else buttons[(i+(e.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length]?.focus();}};
  $('model-cancel').onclick=()=>{closeMenus();$('model-trigger').focus();};
  function groups(){return Object.entries(familyNames).map(([id,name])=>({id,name,models:(state?.models?.models||[]).filter(m=>familyId(m)===id).sort((a,b)=>(modelVersionRank[modelVersions[b.id]]||0)-(modelVersionRank[modelVersions[a.id]]||0))})).filter(g=>g.models.length);}
  function updateModelSelection(){
    const models=state?.models,chosen=models?.models?.find(m=>m.id===modelDraft);
    for(const b of $('model-configs').children){b.setAttribute('aria-checked',String(b.dataset.model===modelDraft));b.classList.toggle('selected',b.dataset.model===modelDraft);}
    const stale=modelDraftRevision!==models?.revision;
    $('model-selection').textContent=stale?'配置已更新，请重新选择':chosen?modelLabel(chosen):'请选择一个配置';
    $('model-selection-details').disabled=!chosen;
    $('model-apply').textContent=modelApplying||models?.switching?'正在切换…':modelDraft===models?.selected?'当前使用中':'应用此配置';
    $('model-apply').disabled=!chosen?.available||modelDraft===models?.selected||stale||!fresh()||requestBusy||resetBusy||modelApplying||models?.switching||!!state?.operation?.busy;
  }
  function showConfigurations(id){
    const group=groups().find(g=>g.id===id);if(!group)return;modelFamily=id;
    $('model-family-title').textContent=group.name+' · '+group.models.length+' 个配置';
    for(const b of $('model-families').children){b.classList.toggle('active',b.dataset.family===id);b.setAttribute('aria-pressed',String(b.dataset.family===id));}
    $('model-configs').replaceChildren();
    for(const model of group.models){
      const b=node('button');b.type='button';b.setAttribute('role','radio');b.dataset.model=model.id;b.dataset.family=id;b.dataset.available=String(model.available);
      b.disabled=!model.available||!fresh()||requestBusy||resetBusy||modelApplying||state.models.switching;
      const text=node('div',null,'model-card-copy'),heading=node('div',null,'model-card-heading');
      heading.append(node('strong',modelLabel(model)));
      const badge=model.id===state.models.selected?'使用中':model.available?'可选择':'暂不可用';
      const badgeNode=node('span',badge,'model-badge');badgeNode.hidden=model.available&&model.id!==state.models.selected;heading.append(badgeNode);
      text.append(heading);
      if(model.disabled_reason)text.append(node('small',model.disabled_reason,'model-description'));
      const evidence=modelEvidence[model.id];
      if(evidence){
        const numbers=node('div',null,'model-numbers');
        const percentages=evidence.values.filter(([,value])=>value.endsWith('%'));
        if(model.id==='yoloe26_tuned')percentages.unshift(['回放合格率',(82/128*100).toFixed(2)+'%']);
        const values=percentages.length?percentages.slice(0,2):[['评分','待评测']];
        for(const [label,value] of values){const metric=node('span');metric.append(node('span',label+' '),node('strong',value));numbers.append(metric);}
        text.append(numbers);
      }
      b.append(node('span','✓','model-choice-mark'),text);
      b.onclick=()=>{modelDraft=model.id;modelDraftRevision=state.models.revision;updateModelSelection();};
      $('model-configs').append(b);
    }
    updateModelSelection();
  }
  $('model-apply').onclick=async()=>{
    updateModelSelection();if($('model-apply').disabled)return;
    const model_id=modelDraft,revision=modelDraftRevision;modelApplying=true;buttons();
    try{await run('select_model',{model_id,revision});closeMenus();$('model-trigger').focus();toast('正在加载配置，核对通过后自动更新画面。');}
    catch(e){toast(e.message);}
    finally{modelApplying=false;buttons();poll();}
  };
  function renderModels(){
    const models=state?.models;if(!models)return;
    const current=models.models?.find(m=>m.id===models.selected);
    $('model-name').textContent=current?modelLabel(current):models.selected_name||'选择模型';
    $('model-current-details').disabled=!current;
    $('model-trigger').dataset.family=current?familyId(current):'other';
    $('model-subtitle').textContent=models.switching?'正在加载配置…':models.error?'切换未完成，保留原模型':models.motion_model_matches?'执行绑定已匹配':'识别对比配置 · 动作暂停';
    const key=JSON.stringify([models.models,models.selected,models.selected_configuration]);if(key===modelKey)return;modelKey=key;
    $('model-families').replaceChildren();
    for(const group of groups()){const b=node('button');b.type='button';b.dataset.family=group.id;b.append(node('strong',group.name),node('span',String(group.models.length),'model-family-count'));b.onclick=()=>showConfigurations(group.id);$('model-families').append(b);}
    const available=groups();if(!available.some(g=>g.id===modelFamily))modelFamily=available[0]?.id;
    showConfigurations(modelFamily);
  }
  async function run(kind,payload,{quiet=false}={}){
    if(!fresh())throw Error('工作站离线，未提交操作');
    if(!quiet&&requestBusy)throw Error('上一项操作尚未确认');
    if(!quiet){requestBusy=true;buttons();}
    try{
      if(local){const result=await api('/api/local-command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:crypto.randomUUID(),kind,payload})});if(!result.ok)throw Error(result.error||'工位拒绝此操作');return result.data;}
      const job=await api('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind,payload})});
      const end=performance.now()+16000;
      while(performance.now()<end){const d=await api('/api/command/'+job.id);if(d.status==='done'){if(!d.result?.ok)throw Error(d.result?.error||'工作站拒绝此操作');return d.result.data;}if(d.status==='expired')throw Error('操作已过期，未发送；请核对现场后重新操作');if(d.status==='unknown')throw Error('响应未确认，不会自动重复执行，请核对工位状态');await new Promise(r=>setTimeout(r,400));}
      throw Error('操作响应未确认，不会自动重复执行，请核对现场');
    }finally{if(!quiet){requestBusy=false;buttons();}}
  }
  function operationPayload(action){const op=state?.operation;const p={action,owner,material,ready:$('scene-ready').checked,expected_operation_id:op?.operation_id??null};if(['start','preview','arm','voice_arm','home','reset'].includes(action)){p.request_id=crypto.randomUUID();activeRequest=p.request_id;}if(['heartbeat','cancel'].includes(action))p.operation_id=op?.operation_id;if(['start','preview','arm','voice_arm'].includes(action)){if(itemHeightEditor?.enabled){p.height_plan=itemHeightEditor.payload();}else{const h=Number($('object-height').value);if(!h||h<1||h>120)throw Error('请填写 1–120 mm 的物体高度');p.object_height_mm=h;}}if(action==='home')p.home_confirmed=p.ready;if(action==='reset')p.reset_confirmed=p.ready;return p;}
  async function action(name){if(resetBusy&&name!=='stop')return;try{const p=operationPayload(name);if(['start','arm','voice_arm','home','reset'].includes(name)&&!p.ready)throw Error('请先核对并确认现场');$('command-message').textContent='正在提交'+({preview:'预演检查',start:'分拣',stop:'停止',reset:'状态重置',home:'回中位'}[name]||'操作')+'…';if(['start','preview','arm','voice_arm','home'].includes(name)){taskSubmitted={id:p.request_id,at:performance.now()};renderTaskProgress(state?.operation);}const reply=await run('operation',p,{quiet:name==='stop'||name==='cancel'});if(reply?.operation_id&&state){state.operation=reply;progressAt=performance.now();renderTaskProgress(reply);}$('command-message').textContent=name==='stop'?'停止请求已送达，请核对现场。':'操作已送达，状态会自动更新。';if(['start','arm','voice_arm','home','reset'].includes(name))$('scene-ready').checked=false;poll();}catch(e){if(taskSubmitted?.id===activeRequest){taskSubmitted.error=e.message;renderTaskProgress(state?.operation);}if(['start','arm','voice_arm','home','reset'].includes(name))$('scene-ready').checked=false;buttons();$('command-message').textContent=e.message;toast(e.message);}}
  $('preview-button').onclick=()=>action('preview');$('start-button').onclick=()=>action('start');$('stop-button').onclick=()=>action('stop');
  function updateManualReviewControls(){
    const review=state?.operation?.manual_review,button=$('manual-review-open');
    button.hidden=!review?.supported;
    button.disabled=reviewSubmitting||requestBusy||resetBusy||!fresh()||!!state?.operation?.busy||!review?.required||!review?.available;
    button.title=!review?.required?'当前没有待复核任务':review?.available?'核对现场后归档本次待复核':'请先处理停止、设备占用或动作回执不明';
    $('manual-review-submit').disabled=reviewSubmitting||!reviewContext||!$('manual-review-confirmed').checked;
    $('manual-review-close').disabled=reviewSubmitting;
  }
  function reviewError(error){const text=String(error.message||error),messages={manual_review_context_changed:'工位任务已变化，请重新核对后复核。',manual_review_device_still_owned:'设备仍被占用，请等待任务完全退出。',manual_review_not_eligible:'当前保护需要通过原恢复流程处理。',manual_review_scene_confirmation_required:'请先核对机械臂位置、空爪和现场。',operation_busy_or_closed:'任务尚未退出，请等待停稳。'};return Object.entries(messages).find(([key])=>text.includes(key))?.[1]||text;}
  $('manual-review-open').onclick=async()=>{try{
    const op=await api('/api/operation'),review=op?.manual_review;
    if(op?.busy||!review?.available)throw Error('当前任务尚不能单独复核，请核对停止或占用状态。');
    reviewContext={operationId:op.operation_id??null,epoch:review.owner_epoch,requestId:review.request_id};
    $('manual-review-confirmed').checked=false;$('manual-review-message').textContent='复核后归档本次未决记录，原任务结果保留。';
    updateManualReviewControls();$('manual-review-dialog').showModal();
  }catch(e){toast(reviewError(e));}};
  $('manual-review-confirmed').onchange=updateManualReviewControls;
  $('manual-review-close').onclick=()=>{if(!reviewSubmitting){reviewContext=null;$('manual-review-dialog').close();}};
  $('manual-review-submit').onclick=async()=>{
    if(reviewSubmitting||!reviewContext||!$('manual-review-confirmed').checked)return;
    reviewSubmitting=true;updateManualReviewControls();const expected={...reviewContext};
    try{
      const op=await api('/api/operation'),review=op?.manual_review;
      if(op?.busy||!review?.available||(op.operation_id??null)!==expected.operationId||review.owner_epoch!==expected.epoch||review.request_id!==expected.requestId)throw Error('工位任务已变化，请重新核对后复核。');
      const reply=await run('operation',{action:'review',owner,request_id:crypto.randomUUID(),expected_operation_id:expected.operationId,
        review_epoch:expected.epoch,review_request_id:expected.requestId,review_confirmed:true,ready:true});
      if(reply?.result?.status!=='operator_reviewed')throw Error('复核响应未确认，请核对任务状态；不会自动重试。');
      if(state)state.operation=reply;reviewContext=null;$('scene-ready').checked=false;$('manual-review-dialog').close();
      $('command-message').textContent=reply.blockers?.length?'已记录复核，仍有其他保护需要处理。':'已复核，可重新核对工作区后开始。';
      toast('已复核，原任务记录已保留。');poll();
    }catch(e){$('manual-review-confirmed').checked=false;reviewContext=null;$('manual-review-message').textContent=reviewError(e);toast(reviewError(e));}
    finally{reviewSubmitting=false;updateManualReviewControls();buttons();}
  };
  for(const b of document.querySelectorAll('[data-action]'))b.onclick=()=>{closeMenus();action(b.dataset.action);};
  for(const b of document.querySelectorAll('[data-material]'))b.onclick=()=>{material=b.dataset.material;for(const item of document.querySelectorAll('[data-material]'))item.classList.toggle('active',item===b);buttons();};
  $('scene-ready').onchange=buttons;$('object-height').oninput=buttons;
  for(const b of document.querySelectorAll('[data-frame]'))b.onclick=()=>{frameKind=b.dataset.frame;frameKey='';for(const item of document.querySelectorAll('[data-frame]'))item.classList.toggle('active',item===b);frame();};
  function buttons(){updateManualReviewControls();itemHeightEditor?.refresh(state?.operation,loadedVision,$('live-frame').src,resetBusy||requestBusy||!!state?.operation?.busy||!fresh(),$('scene-ready').checked&&!!state?.models?.motion_model_matches&&!(state?.operation?.blockers||[]).length);$('station-reset-open').disabled=resetBusy||!fresh()||!state?.operation;const op=state?.operation,connected=fresh(),matched=!!state?.models?.motion_model_matches&&!state?.models?.switching;const height=itemHeightEditor?.enabled?(itemHeightEditor.valid()?1:0):Number($('object-height').value),ready=$('scene-ready').checked;const blocked=(op?.blockers||[]).length>0||op?.board_ownership?.status==='blocked';$('preview-button').disabled=resetBusy||!connected||!op||op.busy||!matched||requestBusy||height<1||height>120;$('start-button').disabled=resetBusy||!connected||!op||op.busy||blocked||!matched||!ready||height<1||height>120||requestBusy;$('stop-button').disabled=!connected||!op;for(const b of document.querySelectorAll('[data-action]'))b.disabled=resetBusy||!connected||!op||requestBusy||(['home','reset'].includes(b.dataset.action)&&(!ready||op.busy))||(['arm','voice_arm'].includes(b.dataset.action)&&(blocked||!matched||!ready||op.busy));$('model-trigger').disabled=resetBusy||!connected||!state?.models||state.models.switching||requestBusy||modelApplying;for(const b of $('model-configs').children)b.disabled=resetBusy||!connected||requestBusy||state?.models?.switching||modelApplying||b.dataset.available!=='true';updateModelSelection();$('image-detect').disabled=resetBusy||!connected||requestBusy||!$('image-file').files?.length;}
  function renderGraspPoints(){
    const svg=$('grasp-overlay'),vision=loadedVision,size=vision?.image_size;
    svg.replaceChildren();svg.hidden=true;svg.setAttribute('hidden','');
    if(!fresh()||frameKind!=='overlay'||$('live-frame').hidden||!Array.isArray(size)||size.length!==2||!size.every(v=>Number.isFinite(v)&&v>0))return;
    svg.setAttribute('viewBox','0 0 '+size.join(' '));svg.setAttribute('preserveAspectRatio','xMidYMid meet');
    const points=(vision.display_grasp_points||[]).filter(item=>Array.isArray(item.pixel)&&item.pixel.length===2&&item.pixel.every((v,i)=>Number.isFinite(v)&&v>=0&&v<size[i]));
    const ns='http://www.w3.org/2000/svg';
    for(const [i,item] of points.slice(0,12).entries()){
      const [x,y]=item.pixel,g=document.createElementNS(ns,'g');
      for(const [x1,y1,x2,y2] of [[x-9,y,x+9,y],[x,y-9,x,y+9]]){const line=document.createElementNS(ns,'line');for(const [k,v] of Object.entries({x1,y1,x2,y2,class:'grasp-cross'}))line.setAttribute(k,v);g.append(line);}
      const c=document.createElementNS(ns,'circle');for(const [k,v] of Object.entries({cx:x,cy:y,r:6,class:'grasp-ring'}))c.setAttribute(k,v);
      const t=document.createElementNS(ns,'text');t.setAttribute('x',Math.max(5,Math.min(size[0]-90,x+13)));t.setAttribute('y',Math.max(18,y-12));t.textContent='抓点 '+(i+1);t.setAttribute('class','grasp-label');g.append(c,t);svg.append(g);
    }
    svg.hidden=!points.length;if(!svg.hidden)svg.removeAttribute('hidden');
  }
  function taskReason(reason){
    if(!reason)return '';
    if(String(reason).includes('prompt_scene_rgb_changed'))return '本件已在发令前暂停：画面变化超过门槛。请撤出手和工具，等物体、相机和光线稳定后重新预演；不要直接连续点开始。';
    if(String(reason).includes('unified_neighbor_in_open_gripper_footprint'))return '夹爪张开范围内有其他物品，请将物品摆开，留出夹爪张开空间后重新预演。';
    if(String(reason).includes('unified_target_too_close_to_other_object'))return '物品间距不足，请将物品摆开后重新预演。';
    if(String(reason).includes('instance_top_below_existing_grasp_z'))return '物体高度不高于当前夹取平面，现有动作无法夹住；请核对实测高度，薄物需要单独适配。';
    if(String(reason).includes('jaw_target_outside_requested_extension'))return '目标夹持间距低于本次允许的下限，请核对物体宽度。';
    if(String(reason).includes('jaw_target_outside_measured_range'))return '目标夹持间距超出实测标定范围；薄物需要单独适配夹爪参数。';
    if(String(reason).includes('instance_count_changed'))return '动作前的工作区物体数量变化，请核对当前画面并重新登记高度。';
    if(String(reason).includes('prompted_batch_requires_instance_heights'))return '整批分拣前，请登记当前画面并填写每件物体的实测高度。';
    if(String(reason).includes('prompted_batch_initial_route_rejected'))return '本批有目标无法规划，尚未开始动作；请查看任务记录。';
    if(String(reason).includes('neutral_return_calibration_jaw_not_open'))return '夹爪仍处于标定闭合位置，请先按受控流程张开空爪，再回初始。';
    if(String(reason).includes('instance_height_missing_or_moved'))return '高度登记与当前识别不匹配，请重新登记画面，并为源区每件物体填写高度。';
    const text=String(reason),messages={unified_target_crosses_bin_boundary:'目标跨过分区边界，请将物体完整放在源区后重试。',board_visual_markers_changed:'定位标记或相机身份检查未通过，请核对工位。',operator_stop:'已停止，请核对现场。',page_disconnected:'控制连接中断，请核对现场。',unified_verification_window_expired:'结果核验超时，请查看任务记录并核对目标箱。'};
    for(const [key,message] of Object.entries(messages))if(text.includes(key))return message;
    return '本次未通过，请在任务记录中查看原因并核对现场。';
  }
  // A phase total is never divided by a kernel-only time or a target budget.
  function cpuComparison(op,stage){
    const record=op?.result?.performance_comparison;
    const row=record?.stages?.[stage];
    const hash=v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v);
    if(!op?.operation_id||record?.schema!=='cpu-fpga-phase-v1'||record.operation_id!==op.operation_id||
       record.configuration_sha256!==op.result?.config_sha256||!hash(record.configuration_sha256)||
       !['pynq_ps_cpu','pc_cpu'].includes(record.cpu_device)||!hash(record.pl_assets_sha256)||
       !row||row.scope!=='phase_end_to_end'||row.includes_transfer!==true||row.outputs_equal!==true||
       !hash(row.cpu_input_sha256)||row.cpu_input_sha256!==row.fpga_input_sha256||
       !Number.isFinite(row.cpu_ms)||!Number.isFinite(row.fpga_ms)||row.cpu_ms<=0||row.fpga_ms<=0)return null;
    const ratio=row.cpu_ms/row.fpga_ms;
    if(!Number.isFinite(ratio)||ratio<=0)return null;
    return {...row,ratio,device:record.cpu_device==='pynq_ps_cpu'?'PYNQ ARM CPU':'电脑 CPU'};
  }
  function comparisonText(row){
    if(!row)return 'CPU 对比 · 待测';
    return (row.ratio>1?'加速比 ':row.ratio<1?'速度比 ':'速度持平 ')+row.ratio.toFixed(2)+'×';
  }
  function renderCpuComparisons(op){
    for(const stage of ['checking','verifying']){
      const button=$('cpu-compare-'+stage),row=cpuComparison(op,stage);
      button.textContent=comparisonText(row);
      button.dataset.measured=String(!!row);
      button.title=row?row.device+' '+(row.cpu_ms/1000).toFixed(3)+' s / FPGA 方案 '+(row.fpga_ms/1000).toFixed(3)+' s；点击查看口径':'尚无同口径 CPU / FPGA 实测；点击查看';
    }
  }
  for(const stage of ['checking','verifying'])$('cpu-compare-'+stage).onclick=()=>{
    const op=state?.operation,row=cpuComparison(op,stage),names={checking:'识别与规划',verifying:'结果核验'};
    $('cpu-comparison-title').textContent=names[stage]+' · CPU 对比';
    const detail=$('cpu-comparison-detail');detail.replaceChildren();
    const facts=node('div',null,'task-detail-facts');
    const items=row?[[row.device+' 基线',(row.cpu_ms/1000).toFixed(3)+' s'],['FPGA 方案（含传输）',(row.fpga_ms/1000).toFixed(3)+' s'],['速度比',row.ratio.toFixed(2)+'×'],['耗时变化',(row.ratio>=1?'减少 ':'增加 ')+(Math.abs(1-row.fpga_ms/row.cpu_ms)*100).toFixed(1)+'%']]:[['CPU 基线','待实测'],['FPGA 方案对照','待实测'],['加速比','暂不可计算']];
    for(const [label,value] of items){const line=node('div');line.append(node('span',label),node('strong',value));facts.append(line);}
    detail.append(facts,node('p',row?'本任务的同输入阶段对照；不代表整轮机械分拣加速。':'当前阶段用时包含取图、识别和检查，不能当作纯 CPU 基线。5 秒 / 2 秒仍是优化目标，尚无对应加速实测。','help-note'));
    if(stage==='checking'){
      detail.append(node('h3','路线计算局部实测 · 10月31日'),node('p','三类各5次，相同输入的完整路线计算；输出指令一致，含本地服务往返，不含采图、识别或机械动作。','help-note'));
      const measured=node('div',null,'task-detail-facts');
      for(const [label,value] of [['PYNQ ARM CPU · 原算法','591.44 ms'],['现役 FPGA 协同','165.22 ms'],['局部速度比','3.58×']]){const line=node('div');line.append(node('span',label),node('strong',value));measured.append(line);}
      detail.append(measured,node('p','3.58×相对于原CPU循环。另测单独高度检查：优化CPU数组7.88 ms，FPGA完整调用34.46 ms；数据往返仍有开销。以上均非本任务完整阶段加速，阶段倍数仍待测。','help-note'));
    }
    if(stage==='verifying'){
      detail.append(node('h3','局部处理实测 · 10月31日'),node('p','目标箱 212 × 168 像素，仅差分与连通块；固定现场画面的无变化样本，3次计时中位数。','help-note'));
      const measured=node('div',null,'task-detail-facts');
      for(const [label,value] of [['PYNQ ARM CPU','18.98 ms'],['FPGA + ARM 连通块','31.41 ms'],['局部速度比','0.60×（变慢）']]){const line=node('div');line.append(node('span',label),node('strong',value));measured.append(line);}
      detail.append(measured,node('p','含打包、缓存同步、硬件等待及读回；不含采图、传输或亮度校正。此候选尚未接入任务，当前阶段仍待完整对照。','help-note'));
      detail.append(node('h3','颜色计算优化 · CPU'),node('p','同一实图与颜色参数，9次交替计时，输出像素完全一致；仅颜色校正。','help-note'));
      const color=node('div',null,'task-detail-facts');
      for(const [label,value] of [['原浮点颜色校正','462.03 ms'],['查找表颜色校正','35.23 ms'],['局部计算速度比','13.11×']]){const line=node('div');line.append(node('span',label),node('strong',value));color.append(line);}
      detail.append(color,node('p','查找表优化已加载，FPGA内核保持原版本。此为CPU计算改进，不代表完整终检或FPGA加速倍数。','help-note'));
    }
    $('cpu-comparison-dialog').showModal();
  };
  function renderTaskProgress(op){
    renderCpuComparisons(op);
    const pending=taskSubmitted&&(!op?.operation_id||op.operation_id!==taskSubmitted.id);
    const p=!pending&&op?.progress&&op.progress.operation_id===op.operation_id?op.progress:null;
    const id=pending?taskSubmitted.id:op?.operation_id||'';
    $('activity-title').hidden=!!id;$('task-progress-title').hidden=!id;$('task-progress-elapsed').hidden=!id;
    $('activity-card').dataset.hasTask=String(!!id);
    $('activity-detail').hidden=!!id||!!op&&(!op.busy&&!(op.blockers||[]).length&&op.board_ownership?.status!=='blocked');$('activity-id').hidden=!id;
    if(!id){$('task-progress').hidden=true;return;}
    if(id!==taskOperationId){taskOperationId=id;taskStartedAt=pending?taskSubmitted.at:performance.now();taskEndedAt=null;taskFinishedElapsed=null;}
    const ended=!pending&&!op?.busy,now=performance.now();if(ended&&taskEndedAt===null){taskEndedAt=now;taskFinishedElapsed=p?.elapsed_s??null;}
    const jawAdjusted=op?.result?.status==='jaw_calibration_command_completed';
    const success=jawAdjusted||['completed_visual_evidence','completed','neutral_return_command_completed'].includes(op?.result?.status);
    const preview=op?.result?.status==='preview_only';
    const needsReview=!!op?.manual_review?.required||op?.board_ownership?.status==='blocked'||!!op?.blockers?.length||op?.result?.status==='needs_review';
    const notStarted=op?.result?.status==='rejected'&&op.result.physical_motion_attempted===false&&op.result.board_uart_write_attempted===false;
    const phase=ended?'ended':p?.phase||(pending?'submitted':'checking');
    const stages=['submitted','checking','connecting','ready','moving','verifying','ended'],index=stages.indexOf(phase);
    const elapsed=p?Math.max(0,ended?(taskFinishedElapsed??p.elapsed_s):p.elapsed_s+(p.terminal||!fresh()?0:(now-progressAt)/1000)):pending||(!ended&&taskSubmitted?.id===id)?Math.max(0,((taskEndedAt??now)-taskStartedAt)/1000):null;
    const events=p?.events||[],seen=new Map(events.map(e=>[e.phase,e.elapsed_s]));
    $('task-progress').hidden=false;$('task-progress').dataset.outcome=ended?(success?'success':preview?'preview':'review'):'active';$('activity-card').dataset.outcome=$('task-progress').dataset.outcome;
    for(const el of document.querySelectorAll('[data-progress]')){
      const stage=el.dataset.progress,i=stages.indexOf(stage),observed=seen.has(stage),next=events.find(e=>e.elapsed_s>=seen.get(stage)&&e.phase!==stage&&stages.indexOf(e.phase)>i);
      el.classList.toggle('active',phase===stage);el.classList.toggle('done',observed&&i<index);el.setAttribute('aria-current',phase===stage?'step':'false');
      const duration=observed?(next?next.elapsed_s-seen.get(stage):elapsed==null?null:elapsed-seen.get(stage)):null;
      const timing=el.querySelector('small');timing.hidden=stage==='submitted'||stage==='ended';timing.textContent=timing.hidden?'':duration!=null?Math.max(0,duration).toFixed(1)+' s':'—';
    }
    const titles={submitted:'正在下达任务',checking:'识别与规划',connecting:'连接机械臂',ready:'准备动作',moving:'抓放与回位',verifying:'结果核验中',ended:jawAdjusted?'夹爪调整完成':success?'本次任务完成':preview?'预演检查完成':needsReview?'已结束 · 待复核':notStarted?'检查未通过 · 未执行动作':'本次已结束 · 未通过'};
    if(op?.batch_mode){titles.moving='整批抓放与回位';titles.verifying='整批统一核验';}
    $('task-progress-title').textContent=fresh()?titles[phase]||'任务处理中':'连接中断 · 进度待同步';
    $('task-progress-elapsed').textContent=elapsed==null?'用时未记录':'总用时 '+elapsed.toFixed(1)+' s'+(!p?' · 本页计时':'');
    const completedIndex=ended&&!success?Math.max(0,...events.filter(e=>e.phase!=='ended').map(e=>stages.indexOf(e.phase))):index;
    const fraction=phase==='moving'&&p?.total_commands?p.completed_commands/p.total_commands:0;
    $('task-progress-fill').style.width=(ended&&success?100:Math.min(96,Math.max(3,(completedIndex+fraction)/6*100)))+'%';
    const names={home:'准备姿态',source_hover:'移至物体上方',source_grasp:'下降抓取',close:'闭合夹爪',release:'松爪放置',neutral:'返回初始位'};
    $('task-progress-note').textContent=!fresh()?'显示最后一次回报，不据计时推定动作完成。':pending?(taskSubmitted.error||'等待工位接收，不会重复发送任务。'):ended?(taskReason(op?.result?.reason||op?.reason)||(success?'结果已由工位回报。':preview?'未执行抓放。':'请在任务记录中核对原始结果。')):phase==='connecting'||phase==='ready'?'此处可能听到控制器提示音；尚无声音回报，不表示抓放或核验完成。':phase==='moving'&&p?.total_commands?(names[p.command_name]||p.command_name||'执行路线')+' · 回显与等待 '+p.completed_commands+' / '+p.total_commands+' 步':p?'按工位实际节点更新，进度长度表示流程节点。':'工位尚未提供细分节点，等待真实回报。';
    $('task-progress-note').hidden=!!fresh()&&!pending&&(ended?success||preview:!['connecting','ready','moving'].includes(phase));
    if(pending&&!taskSubmitted.error)$('task-progress-note').hidden=true;
  }
  function frame(){if(!fresh()||!state?.vision){$('live-frame').hidden=true;$('grasp-overlay').hidden=true;$('grasp-overlay').setAttribute('hidden','');loadedVision=null;pendingFrame=null;$('frame-empty').hidden=false;$('frame-age').hidden=true;return;}const vision=state.vision,key=[frameKind,vision.frame_key||vision.frame_id,state.received_at].join(':');if(key===frameKey)return;frameKey=key;pendingFrame={vision:structuredClone(vision),kind:frameKind};$('grasp-overlay').hidden=true;$('grasp-overlay').setAttribute('hidden','');loadedVision=null;$('live-frame').src='/api/frame?kind='+frameKind+'&key='+encodeURIComponent(vision.frame_key||'')+'&v='+encodeURIComponent(key);}
  $('live-frame').onload=()=>{if(!fresh()||!pendingFrame)return;$('live-frame').hidden=false;loadedVision=pendingFrame.kind==='overlay'?pendingFrame.vision:null;renderGraspPoints();$('frame-empty').hidden=true;$('frame-age').hidden=false;$('frame-age').textContent='实时同步 · '+new Date().toLocaleTimeString('zh-CN',{hour12:false,timeZone:'Asia/Shanghai'});};
  $('live-frame').onerror=()=>{$('live-frame').hidden=true;$('grasp-overlay').hidden=true;$('grasp-overlay').setAttribute('hidden','');loadedVision=null;pendingFrame=null;frameKey='';$('frame-empty').hidden=false;$('frame-empty-detail').textContent='画面暂不可用，等待新的工位图像。';$('frame-age').hidden=true;};
  function operationText(op){if(op&&!state?.models?.motion_model_matches)return ['当前为识别配置','此配置用于识别对比；动作需切回已绑定的执行配置并通过现场检查。','仅识别'];if(!op)return ['控制状态不可读','等待板端服务恢复，动作入口保持禁用。','状态未知'];if(op.busy)return [{checking_preview:'正在预演',checking:'正在检查',waiting_clear:'等待清空源区',standby:'等待放入物品',voice_waiting:'等待语音口令',executing:'分拣处理中',stopping:'正在停止'}[op.state]||'正在处理','执行与状态由工位控制器管理。','处理中'];if(op.board_ownership?.status==='blocked'||(op.blockers||[]).length)return ['需要复核','上一任务或现场检查尚未通过，请查看任务记录并核对现场。','待复核'];if(op.state==='finished')return ['本次已结束','结果按原始任务记录展示。','已结束'];return ['可以准备作业','选择类别，填写高度，再核对现场。','空闲'];}
  function render(){renderModels();const connected=fresh(),op=state?.operation;const [title,detail,pill]=operationText(op);$('connection-badge').classList.toggle('online',connected);$('connection-badge').querySelector('span').textContent=connected?'工作站在线':'工作站离线';$('sidebar-dot').classList.toggle('online',connected);$('camera-dot').classList.toggle('online',connected&&!!state?.vision);$('camera-state').textContent=connected?'工位实时画面':'等待工位连接';$('activity-title').textContent=connected?title:'工位连接中断';$('activity-detail').textContent=connected?detail:'画面与动作同步已暂停，连接恢复后自动更新。';$('operation-state').textContent=connected?pill:'离线';$('activity-symbol').textContent=op?.busy?'↻':pill==='待复核'?'◇':'◌';$('activity-material').textContent=labels[op?.material_selection||material]||'全部分类';$('activity-id').textContent=op?.operation_id?op.operation_id.slice(0,8):'—';$('device-state').textContent=!connected?'状态未知':op?'PYNQ · '+(op.board_ownership?.status==='blocked'?'待复核':'已连接'):'板端连接待恢复';$('sync-time').textContent=connected?'最近同步 · '+new Date().toLocaleTimeString('zh-CN',{hour12:false,timeZone:'Asia/Shanghai'}):'等待状态同步';const ex=state?.health?.execution;$('compute-label').textContent=ex?.device==='cuda:0'?'ROCm GPU · '+(ex.precision||'FP16').toUpperCase():'视觉工作站 · PYNQ 控制';$('settings-gpu').textContent=ex?((ex.device_name||ex.device)+' · '+(ex.precision||'').toUpperCase()):'未同步';$('settings-connection').textContent=connected?(local?'已直连本机工位，画面与状态自动更新。':'本机连接器已连接，正在同步工位状态。'):(local?'本机工位服务暂不可用，请检查原工作台状态。':'本机连接器离线，或电脑未启动。');$('settings-status').textContent=connected?'已连接':'离线';$('inference-time').textContent=state?.vision?.inference_ms!=null?Math.round(state.vision.inference_ms)+' ms':'— ms';const metrics=state?.vision;$('task-performance').textContent='识别 '+(Number.isFinite(metrics?.inference_ms)?metrics.inference_ms.toFixed(0)+' ms':'—')+' · 抓点 '+(Number.isFinite(metrics?.display_geometry_timing?.elapsed_ms)?metrics.display_geometry_timing.elapsed_ms.toFixed(1)+' ms':'—');$('detection-chips').replaceChildren();const instances=state?.vision?.instances||[];if(!instances.length)$('detection-chips').append(node('span',connected?'当前未检出目标':'识别结果将在这里显示','subtle'));for(const item of instances.slice(0,5)){const chip=node('span',(item.official_label||'未提供细分类')+' · '+(labels[item.label]||item.label)+' '+Math.round(item.confidence*100)+'%','detection-chip');const material=String(item.label||'').toLowerCase();if(['plastic','metal','paper'].includes(material))chip.dataset.material=material;$('detection-chips').append(chip);}history();buttons();renderTaskProgress(op);frame();}
  const statusName=value=>({jaw_calibration_command_completed:'夹爪调整已完成',motion_completed_awaiting_batch_verification:'待整批核验',neutral_return_command_completed:'回中位已执行',completed_visual_evidence:'视觉核验通过',completed:'已完成',needs_review:'待复核',rejected:'未通过',preview_only:'预演',failed:'未通过',error:'未通过',verification_failed:'核验未通过'}[value]||value||'记录');
  function history(){const records=state?.tasks?.records||[];const signature=JSON.stringify(records);if(recordsKey===signature)return;recordsKey=signature;$('history-count').textContent=records.length||'—';$('recent-tasks').replaceChildren();$('history-rows').replaceChildren();$('history-empty').hidden=records.length>0;if(!records.length)$('history-empty').textContent=state?.tasks?'暂无任务记录':'工位记录暂不可读，稍后自动更新';if(!records.length)$('recent-tasks').append(node('p',(local?'工位记录暂未读到':'连接工位后自动同步'),'sidebar-muted'));for(const [index,record] of records.entries()){const recent=node('button',null,'recent-task');recent.append(node('span',(labels[record.label]||'单件')+' · '+statusName(record.raw_status)),node('small',clock(record.recorded_at)));recent.onclick=()=>task(record.task_id);if(index<6)$('recent-tasks').append(recent);const tr=node('tr');tr.append(node('td',record.task_id?.slice(0,8)||'—'),node('td',labels[record.label]||'—'),node('td',statusName(record.raw_status)),node('td',clock(record.recorded_at)));const cell=node('td'),button=node('button','查看 ↗');button.onclick=()=>task(record.task_id);cell.append(button);tr.append(cell);$('history-rows').append(tr);}}
  async function task(id){$('task-detail').replaceChildren(node('p','正在读取原始任务记录…','subtle'));$('task-dialog').showModal();try{const data=state?.tasks?.task_id===id?state.tasks:await run('task',{task_id:id});if(!data?.available)throw Error('当前没有可读取的完整记录');const facts=node('div',null,'task-detail-facts');for(const [label,value] of [['任务',data.task_id],['类别',labels[data.label]||data.label],['原始结果',statusName(data.raw_status||data.status)],['整轮耗时',data.elapsed_s!=null?data.elapsed_s.toFixed(1)+' 秒':'未记录'],['命令回显',(data.echoed_commands??0)+' / '+(data.planned_commands??0)],['完成等待',String(data.waited_commands??0)]]){const row=node('div');row.append(node('span',label),node('strong',value||'—'));facts.append(row);}$('task-detail').replaceChildren(facts);if(data.reason)$('task-detail').append(node('p',data.reason,'task-detail-message'));for(const item of data.commands||[]){const row=node('div',null,'command-step');row.append(node('span',item.index+'. '+(item.step||'步骤')),node('span',item.completed_wait?'已完成等待':item.state||'未记录'));$('task-detail').append(row);}}catch(e){$('task-detail').replaceChildren(node('p',e.message,'task-detail-message'));}}
  $('upload-open').onclick=()=>$('upload-dialog').showModal();$('image-file').onchange=buttons;
  $('image-detect').onclick=async()=>{const file=$('image-file').files[0];if(!file)return;if(file.size>1500000){toast('请选择不超过 1.5 MB 的图片');return;}try{$('upload-message').textContent='正在使用当前模型识别…';const base64=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.onerror=reject;r.readAsDataURL(file);});const data=await run('detect',{image_base64:base64});if(data.overlay_base64){$('upload-result').src='data:image/jpeg;base64,'+data.overlay_base64;$('upload-result').hidden=false;}$('upload-message').textContent='识别完成 · '+(data.instances?.length||0)+' 个目标，不触发机械动作。';}catch(e){$('upload-message').textContent=e.message;}};
  let polling=false;
  async function poll(){if(polling)return;polling=true;try{const data=await api('/api/state');online=data.online;state=data.snapshot;progressAt=performance.now();lastPollAt=performance.now();if(taskSubmitted?.id===state?.operation?.operation_id)taskSubmitted=null;baseAge=data.device_age_ms??Infinity;render();if(!local&&fresh()&&itemHeightEditor?.library&&!itemHeightEditor.profileLoaded&&!itemHeightEditor.profileLoading){itemHeightEditor.profileLoading=true;itemHeightEditor.library.load().then(()=>{itemHeightEditor.profileLoaded=true;itemHeightEditor.autoKey='';itemHeightEditor.renderLibrary();}).catch(()=>{}).finally(()=>{itemHeightEditor.profileLoading=false;});}}catch(e){online=false;buttons();frame();$('connection-badge').querySelector('span').textContent='连接中断';$('connection-badge').classList.remove('online');$('command-message').textContent=e.message;}finally{polling=false;}}
  setInterval(()=>{buttons();renderTaskProgress(state?.operation);if(!fresh())frame();},250);
  // Local tab switching keeps this real client's finite request alive. Remote
  // pages still require the foreground; freshness and exact ownership always apply.
  setInterval(async()=>{const op=state?.operation;if(!fresh()||heartbeatBusy||(!local&&document.visibilityState!=='visible')||!op?.busy||op.owner!==owner||op.operation_id!==activeRequest)return;heartbeatBusy=true;try{await run('operation',{action:'heartbeat',owner,operation_id:op.operation_id},{quiet:true});}catch(e){$('command-message').textContent='本次连接未确认，请核对现场。';}finally{heartbeatBusy=false;}},2500);
  poll();setInterval(poll,1400);
})();

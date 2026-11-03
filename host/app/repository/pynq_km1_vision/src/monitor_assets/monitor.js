'use strict';
// Timing and identity checks shared by the browser and its regression tests.
(function(root){
 const finite=x=>typeof x==='number'&&Number.isFinite(x)&&x>=0;
 const identity=h=>JSON.stringify([h?.model_sha256,h?.runtime_id,h?.execution?.device,h?.execution?.precision]);
 function fresh(d,h){
  return !!(d&&h?.live_enabled&&h.camera_connected&&h.execution?.device==='cuda:0'&&h.execution.hip&&
   typeof h.model_sha256==='string'&&h.model_sha256&&typeof h.runtime_id==='string'&&h.runtime_id&&
   d.model_version===h.model_sha256&&d.runtime_id===h.runtime_id&&
   finite(d.result_age_ms)&&d.result_age_ms<2000&&finite(d.inference_ms));
 }
 function geometry(d){
  const t=d?.display_geometry_timing;
  if(!t||t.schema_version!=='monitor-display-geometry-timing-v1'||t.backend!=='pc_cpu_opencv_numpy'||
   t.scope!=='display_candidates_all_instances'||t.control_authorized!==false||t.calibrated!==false||
   !['frame_id','source_frame_id','source_stream_id','model_version','runtime_id'].every(k=>t[k]===d[k])||
   !['elapsed_ms','mask_decode_ms','point_compute_ms'].every(k=>finite(t[k]))||
   t.mask_decode_ms+t.point_compute_ms>t.elapsed_ms+.001||
   !Number.isInteger(t.point_count)||!Number.isInteger(t.decoded_masks)||t.point_count<0||
   t.decoded_masks<t.point_count||t.decoded_masks>Math.min(d.instances?.length??0,20)||
   t.point_count!==(d.display_grasp_points?.length??0))return null;
  return t;
 }
 function baselineMatches(b,h){
  return !!(b?.available&&b.source==='offline_benchmark'&&b.live_speedup===false&&
   h?.execution?.device==='cuda:0'&&h.execution.hip&&b.model_sha256===h.model_sha256&&
   b.gpu_precision===h.execution.precision&&
   ['cpu_median_ms','gpu_median_ms','median_speedup','batch_speedup','same_fp32_median_speedup']
    .every(k=>finite(b[k])&&b[k]>0));
 }
 class SampleWindow{
  constructor(limit=30){this.limit=limit;this.clear();}
  clear(){this.rows=[];this.key=null;this.identity=null;}
  add(d,h){
   if(!fresh(d,h)){this.clear();return null;}
   const current=identity(h);
   if(this.identity!==current){this.clear();this.identity=current;}
   const key=JSON.stringify([d.source_stream_id,d.frame_id]);
   if(this.rows.length&&this.rows[this.rows.length-1].stream!==d.source_stream_id)this.rows=[];
   if(key!==this.key){this.key=key;this.rows.push({ms:d.inference_ms,stream:d.source_stream_id});if(this.rows.length>this.limit)this.rows.shift();}
   const values=this.rows.map(x=>x.ms).sort((a,b)=>a-b);
   const quantile=p=>{const pos=(values.length-1)*p,low=Math.floor(pos),high=Math.ceil(pos);return values[low]+(values[high]-values[low])*(pos-low);};
   return {samples:values.length,median_ms:quantile(.5),p95_ms:quantile(.95)};
  }
 }
 function assetsMatch(a,b){
  const keys=['station_vision.bit','station_vision.hwh'];
  return !!(a&&b&&keys.every(k=>typeof a[k]==='string'&&/^[a-f0-9]{64}$/.test(a[k])&&a[k]===b[k]));
 }
 function boardSample(d){
  const e=d?.evidence;
  if(!d?.available||!d.configured||!d.camera_connected||d.faulted||d.backend!=='pl_board_local'||
   d.contract!=='station-vision-v1'||!assetsMatch(d.assets,d.assets)||
   typeof d.stream_id!=='string'||!d.stream_id||typeof d.config_sha256!=='string'||!d.config_sha256||
   !finite(d.frame_age_ms)||d.frame_age_ms>=2000||!e||e.backend!=='pl_board_local'||
   e.config_sha256!==d.config_sha256||e.automatic_motion_allowed!==false||
   !Number.isInteger(e.frame_id)||e.frame_id<0||e.frame_id!==d.frame_id||
   !Number.isInteger(e.baseline_frame_id)||e.baseline_frame_id<0||e.baseline_frame_id!==d.baseline_frame_id||
   !finite(e.processing_ms)||e.processing_ms<=0)return null;
  return e;
 }
 function boardBenchmarkMatches(b,d){
  return !!(b?.available&&b.source==='saved_board_startup_benchmark'&&b.live_speedup===false&&
   b.whole_system_speedup_measured===false&&assetsMatch(b.assets,b.assets)&&
   (!d?.assets||assetsMatch(b.assets,d.assets))&&
   (!d?.configured||d.config_sha256===b.config_sha256||d.runtime_regions_observed===true)&&
   ['cpu_median_ms','pl_median_ms','median_speedup'].every(k=>finite(b[k])&&b[k]>0)&&
   Number.isInteger(b.samples)&&b.samples>0);
 }
 class BoardWindow{
  constructor(limit=30){this.limit=limit;this.clear();}
  clear(){this.rows=[];this.key=null;this.identity=null;}
  add(d){
   const e=boardSample(d);if(!e){this.clear();return null;}
   const current=JSON.stringify([d.stream_id,d.config_sha256,d.assets,e.baseline_frame_id]);
   if(this.identity!==current){this.clear();this.identity=current;}
   if(this.key!==e.frame_id){this.key=e.frame_id;this.rows.push(e.processing_ms);if(this.rows.length>this.limit)this.rows.shift();}
   const values=[...this.rows].sort((a,b)=>a-b);
   const quantile=p=>{const pos=(values.length-1)*p,low=Math.floor(pos),high=Math.ceil(pos);return values[low]+(values[high]-values[low])*(pos-low);};
   return {samples:values.length,median_ms:quantile(.5),p95_ms:quantile(.95)};
  }
 }
 function operationFailure(reason,checks=[]){
  if(typeof reason!=='string'||!reason)return '';
  if(reason.includes('unified_no_free_verified_bin_slot')){
   const names={metal:'金属箱',plastic:'塑料箱',paper:'纸类箱'};
   const check=checks.find(c=>c.reason!=='free');
   if(check?.occupied_px>0)return (names[check.label]||'目标箱')+'落点检测到物体，不能重复投放。';
   if(check?.reason==='empty_reference_mismatch')return (names[check.label]||'目标箱')+'与空箱参考差异 '+(check.changed_fraction*100).toFixed(1)+'%，允许 '+(check.max_changed_fraction*100).toFixed(1)+'%；请核对空箱参考。';
   return '目标箱没有空闲的已测落点；重置状态不会改变箱内占用或空箱参考。';
  }
  if(reason.includes('unified_known_profile_dimensions'))return '识别出的目标尺寸不符合该类别的已测样件，请核对类别及物品朝向。';
  if(reason.includes('neutral_return_'))return '回中位条件未通过：需要空夹爪和可信的准备位/中位指令记录，请查看任务记录；不要从不明位置强行回位。';
  if(reason.includes('unified_fragmented_specimen'))return '目标轮廓分成多块，未形成可靠抓点；请让目标完整露出并与其他物品分开。';
  if(reason.includes('prompt_selected_target_missing_or_moved'))return '选中的目标在最新帧中缺失、置信度不足或位置变化，已停止下发动作；请核对实时识别和遮挡。这不是控制断连。';
  if(reason.includes('prompt_selected_target_ambiguous'))return '选中的目标出现多个匹配轮廓，已停止下发动作，请分开摆放。';
  if(reason.includes('unified_specimen_area'))return '目标轮廓面积不在旧样件模板范围；这不是连接超时。'+reason;
  if(reason.includes('unified_object_exceeds_gripper'))return '当前物体实测宽度超过夹爪允许开度，未发送动作。'+reason;
  if(reason.includes('unified_known_profile_dimensions'))return '实测尺寸与旧样件模板不符，未发送动作。'+reason;
  if(reason.includes('instance_top_below_existing_grasp_z'))return '物体高度不高于当前夹取平面，现有动作无法夹住；薄物需要单独适配。';
  if(reason.includes('jaw_target_outside_requested_extension'))return '目标夹持间距低于本次允许的下限，请核对物体宽度。';
  if(reason.includes('jaw_target_outside_measured_range'))return '目标夹持间距超出实测标定范围；薄物需要单独适配夹爪参数。';
  if(reason.includes('unified_no_pickable_instance'))return '没有找到可安全抓取的目标，请查看预演结果和任务记录。';
  if(reason.includes('motion_requires_two_fixed_markers'))return '当前不足2枚已登记固定标记，动作前检查停止；请让至少2枚标记完整露出。';
  if(reason.includes('motion_requires_three_fixed_markers'))return '当时不足3枚固定标记，动作前检查停止；原记录保留，请查看当前标记门槛。';
  if(reason.includes('fixed_markers_moved_over_half_pixel'))return '初始三帧定位未通过：标记角点变化超过0.5像素，本次没有进入抓取动作。运动中的5像素限制是另一项检查；重置不会消除定位抖动。';
  if(reason.includes('fixed_markers_moved_over_initial_limit'))return '初始三帧定位未通过：标记角点变化超过当前定位位移上限，本次没有进入抓取动作；请保持相机和标记固定。';
  if(reason.includes('relocation_observation_window_expired'))return '初始定位采图超过当前窗口，本次没有进入抓取动作；请核对相机采图延迟。';
  if(reason.includes('motion_static_marker_moved_occluded_or_ambiguous'))return '固定标记图块的一致性核验未通过，可能有遮挡、亮度变化或位移；请查看任务记录区分已完成动作与最终视觉核验。';
  if(reason.includes('station_unexpected_region_change:static_'))return '固定标记参考区的图像变化超限；抓放动作与视觉核验结果分别记录，请查看任务记录。';
 if(reason.includes('fixed_marker_task_points_unstable'))return '初始三帧定位未通过：换算到工作区的坐标变化超过当前允许上限，本次没有进入抓取动作；请核对标记清晰度和相机固定状态。';
 if(reason.includes('marker_reprojection_error'))return '固定标记定位误差未通过，未下发下一步，请核对标记清晰度和相机固定状态。';
  if(reason.includes('camera_not_stable')||reason.includes('common_fixed_markers'))return '固定标记定位不稳定或被遮挡，请保持相机和桌面固定，让标记完整露出。';
  if(/WinError (5|32|33)/.test(reason))return '动作记录文件暂被占用或被Windows拒绝更新，本轮未完成；这不是识别门槛。请核对机械臂停稳、张爪无夹物后重置，不能直接重复抓取。';
  return '检查未通过，请查看任务记录中的具体原因。';
 }
 function operationActivity(d,{connected=true,pending=false,requestAction='',ready=false}={}){
  const view=(tone,title,detail)=>({tone,title,detail});
  if(!connected)return view('error','控制连接中断 · 状态未知','无法确认机械臂是否停止，请核对现场；连接恢复后自动更新。');
  if(!d)return view('waiting','正在读取工位状态…','等待控制服务回应。');
  if(![1,2,3].includes(d.control_version))return view('error','控制服务版本不匹配','请核对监视服务，当前不能确认执行状态。');
  if(d.reason==='stop_write_failed_inspect_station')return view('error','停止请求写入失败 · 请立即核对现场','无法确认停止请求已送达执行器，请立即现场检查机械臂。');
  if(pending)return view('error','请求结果未确认','请核对现场和任务记录，不要重复点击开始。');
  if(requestAction){
   const labels={preview:'正在提交预演检查',start:'正在提交分拣请求',arm:'正在申请待命',voice_arm:'正在申请语音待命',stop:'正在发送停止请求',cancel:'正在退出待命',reset:'正在重置记录',home:'正在提交回中位请求'};
   return view('running',labels[requestAction]||'正在提交请求','正在等待控制服务确认，请勿重复点击。');
  }
  if(d.busy){
   const countdown=Number.isFinite(d.remaining_s)?' 剩余 '+Math.max(0,d.remaining_s)+' 秒。':'';
   if(d.state==='stopping')return view('error','已请求停止 · 等待执行器','尚未确认机械臂停稳，请现场核对。');
   if(d.state==='checking_preview')return view('running','正在预演检查','采图、定位、识别并检查抓取计划；预演不发送机械指令。');
   if(d.state==='checking')return view('running','正在检查待命条件','核对工位与取物区，检查通过后进入等待放物。'+countdown);
   if(d.state==='waiting_clear')return view('waiting','等待清空取物区','请先移走当前所选类别的待抓物品，确认空区后再放入新的目标。'+countdown);
   if(d.state==='standby')return view('waiting','待命中 · 等待放入物品','放入已测样件并立即撤手，目标稳定且检查通过后开始一次分拣。'+countdown);
   if(d.state==='voice_waiting')return view('waiting','语音待命 · 等待口令','本批许可已开启，说“小爱同学”后说分拣口令。'+countdown);
   if(d.state==='executing'&&d.active_action==='home')return view('running','正在处理回中位','检查通过后低速回位；等待执行器返回结果，请勿触碰机械臂。');
   if(d.state==='executing')return view('running',d.batch_mode?'本批分拣处理中':'单件分拣处理中','执行器正在处理定位、识别、路径检查或抓放核验；当前接口未报告具体机械步骤，请等待结果。');
   return view('running','任务处理中','等待执行器返回当前结果，请勿重复开始。');
  }
  if(!d.configured)return view('error','工位控制尚未就绪','控制档案暂不可用，请查看右侧阻断说明。');
  if(d.blockers?.length)return view('error',d.blockers.includes('operator_stop_requested')?'停止保护已锁存 · 请核对现场':'分拣被阻断 · 需要处理','请查看阻断原因；核对现场后处理记录，再重新开始。');
  const r=d.result;
  if(d.state==='rejected'||['needs_review','rejected','blocked'].includes(r?.status))return view('error','本次未通过 · 请核对记录',operationFailure(r?.reason,r?.slot_checks)||'查看任务记录中的拒绝或异常原因；不会自动重试。');
  if(d.state==='stopped')return view('waiting','本次已退出','请核对现场及任务记录；下一批需要重新确认现场。');
  if(r?.status==='state_reset')return view('waiting','状态已重置 · 等待操作','记录已重置，没有移动机械臂；请重新核对现场。');
  if(r?.status==='neutral_return_command_completed')return view('success','回中位指令已完成','完整回显与等待已结束，请现场核对实际姿态。');
  if(r?.preview||r?.status==='preview_only')return view('success','预演已结束 · 未执行抓取','查看预演记录；确认现场就绪后再开始分拣。');
  if(d.state==='finished'){
   const passed=r?.status==='completed_visual_evidence'||(r?.status==='completed'&&r?.final_verification?.status==='batch_visual_evidence_pass');
   return view(passed?'success':'waiting',passed?'本次视觉核验通过':'本次已结束 · 请查看结果',passed?'视觉结果已保存；下一批需要重新核对现场。':'执行结果已返回，动作记录与视觉核验请分别核对。');
  }
  if(d.state!=='idle')return view('error','控制状态未知','请核对控制服务和任务记录。');
  return view('waiting',ready?'现场已确认 · 等待开始':'分拣空闲 · 等待操作',ready?'请选择开始分拣、开启待命或预演检查。':'可先做预演检查；实际分拣前请勾选右侧“已核对现场”。');
 }
 function createOperationKeepalive({getOperation,owner,send,onAck=()=>{},onFailure=()=>{},schedule=setTimeout,cancel=clearTimeout,clock=Date.now}){
  let closed=false,timer=null,lastAck=0;
  async function tick(){
   if(closed)return;
   const current=getOperation();
   try{
    if(current?.busy&&current.owner===owner){
     const id=current.operation_id;
     const reply=await send({action:'heartbeat',token:current.token,owner,operation_id:id});
     if(reply.operation_id!==id||!(reply.heartbeat_accepted===true||(reply.busy&&reply.owner===owner)))throw Error('heartbeat_not_confirmed');
     if(!closed){lastAck=clock();onAck()}
    }
   }catch(error){if(!closed)onFailure(error)}
   finally{if(!closed)timer=schedule(tick,2000)}
  }
  return {start(){if(timer===null&&!closed)tick()},stop(){closed=true;if(timer!==null)cancel(timer)},recent(){return lastAck>0&&clock()-lastAck<6000}};
 }
 const api={createOperationKeepalive,identity,fresh,geometry,baselineMatches,SampleWindow,boardSample,boardBenchmarkMatches,BoardWindow,operationFailure,operationActivity};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.MonitorMetrics=api;
})(globalThis);

// Dashboard bindings.
'use strict';
const $=id=>document.getElementById(id);
const names={plastic:'塑料',metal:'金属',paper:'纸质'};
const regions={source:'取物区',bin_plastic:'塑料箱',bin_metal:'金属箱',bin_paper:'纸类箱',static_a:'背景 A',static_b:'背景 B'};
const statuses={state_reset:'状态已重置',needs_review:'需复核',completed_visual_evidence:'原始视觉核验通过',preview_only:'仅预演',empty_scene:'未检出取物目标',rejected:'任务拒绝',verifying:'正在核验',motion_intent_reserved:'动作准备中',motion_in_progress:'动作执行记录中',motion_completed_awaiting_batch_verification:'动作完成 · 等待整批核验',completed:'整批视觉核验通过',blocked:'本批已停止',item_limit_reached:'达到本批件数上限',selection_complete:'所选类别处理结束',no_matching_material:'未检出所选类别'};
const stepNames={home:'准备高位',source_hover:'目标上方',source_descend_mid:'下降过渡',source_grip:'抓取高度',close:'闭爪',source_lift_mid:'抬升过渡',source_lift_hover:'抬升至悬停',release:'释放',return:'返回中位'};
const stateNames={planned:'尚未执行',intent_only:'仅发送意图',written:'已有发送 · 回显待核对',echo_matched:'发送与完整回显匹配',identity_mismatch:'指令身份不符 · 需复核'};
let screen='live',focus='result',health=null,liveResult=null,uploadResult=null,station=null;
let resultAt=0,cameraAt=0,cameraUrl=null,lastResultKey=null,taskChoice='',taskData=null,taskSignature=null,recordKey=null;
let modelIdentity=null,workflow=null,uploadBusy=false;
let deviceBenchmark=null,boardBenchmark=null,boardAt=0;
const boardWindow=new MonitorMetrics.BoardWindow(30);
const runtimeWindow=new MonitorMetrics.SampleWindow(30);
const fmt=(x,d=1)=>typeof x==='number'&&Number.isFinite(x)?x.toFixed(d):'—';
const xy=x=>Array.isArray(x)?x.map(v=>fmt(v,2)).join(', '):'—';
const time=x=>x?new Date(x).toLocaleString('zh-CN',{hour12:false}):'未记录时间';
const text=(id,value)=>{$(id).textContent=value??'—'};
let operation=null,operationRequest=false,operationPending=false,operationNotice='';
let operationRequestAction='',operationUpdatedAt=0,operationConnected=true,activityKey='',activitySince=0;
const operationOwner=crypto.randomUUID();
const operationMaterials={AUTO:'全部分类',PAPER:'纸类分拣',METAL:'金属分拣',PLASTIC:'塑料分拣'};
statuses.neutral_return_command_completed='回中位指令完成';
const operationPaths={AUTO:'/operation/all',PAPER:'/operation/paper',METAL:'/operation/metal',PLASTIC:'/operation/plastic'};
const operationNames={idle:'未待命',checking:'检查待命条件',checking_preview:'正在预演',waiting_clear:'请先清空取物区',standby:'待命 · 等待放物',voice_waiting:'语音待命 · 等待口令',executing:'单件执行中',stopping:'已请求停止 · 等待执行器',stopped:'已退出',finished:'本次已结束',rejected:'本次未执行 / 需查看记录'};
const blockerNames={station_entry_unavailable:'工位控制档案尚未就绪',station_evidence_changed:'标定或抓取实现与工位绑定记录不一致，需重新核对',station_binding_or_configuration_unavailable:'工位档案已变化或不可读取',operator_stop_requested:'停止保护已锁存；先点“重置状态（不回位）”核对现场并确认解除，再点“回中位”',arm_in_use_or_stale_device_lease:'设备占用或遗留锁待核对',existing_physical_trial_unresolved:'上一任务尚待复核',unified_batch_needs_review:'上一批次尚待核对处理，请先查看任务记录',station_scene_geometry_needs_review:'工位区域尚待确认',station_empty_bins_reference_needs_confirmation:'空箱基准尚待确认'};
const operationReasons={operator_stop:'已请求停止，恢复前须核对现场和停止记录。',standby_cancelled:'已退出待命，没有安排下一次抓取。',page_disconnected:'页面连接已中断，已退出待命或请求执行器停止。',standby_timeout:'待命已超时，请重新核对现场后开启。',attempt_rejected_check_task_record:'条件未通过或执行异常，请查看任务记录；不会自动重试。',stop_write_failed_inspect_station:'停止请求写入失败，请立即现场检查。',monitor_shutdown:'监视服务已关闭。'};
function renderActivity(){
 const connected=operationConnected&&(!operationUpdatedAt||Date.now()-operationUpdatedAt<6500);
 const a=MonitorMetrics.operationActivity(operation,{connected,pending:operationPending,requestAction:operationRequestAction,ready:$('operation-ready').checked});
 const key=JSON.stringify([operation?.operation_id,operation?.state,operation?.active_action,a.title]);
 if(activityKey!==key){activityKey=key;activitySince=Date.now()}
 $('activity-banner').setAttribute('data-tone',a.tone);
 text('activity-title',a.title);
 let detail=a.detail;
 if(connected&&!operation?.busy&&!operationRequestAction&&!operationPending&&operation?.blockers?.length)detail=operation.blockers.map(x=>blockerNames[x]||'工位资格检查未通过').join('；');
 text('activity-detail',detail);
 const selected=operation?.busy?operation.material_selection:$('operation-material').value;
 text('activity-material','当前入口 · '+(operationMaterials[selected]||'待确认'));
 text('activity-clock',connected&&operation?.busy?'本页已观察当前阶段 '+Math.floor((Date.now()-activitySince)/1000)+' 秒':'');
 text('activity-updated',operationUpdatedAt?(connected?'状态更新于 ':'最后收到状态 ')+new Date(operationUpdatedAt).toLocaleTimeString('zh-CN',{hour12:false}):'尚未收到控制状态');
 if(!connected)for(const id of ['preview','start','arm','voice-arm','cancel','reset','home'])$('operation-'+id).disabled=true;
}
function renderOperation(d){
 if(![1,2,3].includes(d.control_version)){operation=d;renderActivity();for(const id of ['preview','start','arm','voice-arm','cancel','stop','reset','home'])$('operation-'+id).disabled=true;text('operation-state','后端版本待更新');text('operation-message','请使用本轮新版监视服务；旧进程需要正常退出后重新启动。');return}
 operation=d;text('operation-state',operationNames[d.state]||'控制状态未知');
 $('voice-controls').hidden=!d.voice_service?.supported;
 const directVoice=d.voice_service?.direct_commands===true;
 $('operation-voice-arm').hidden=directVoice;
 text('voice-state',d.voice_service?.connected?(directVoice?'语音直控在线':d.voice_service.armed?'本批语音已就绪':'语音在线 · 等待确认'):'语音连接待恢复');
 text('voice-instructions',directVoice?'持续监听：说“小爱同学”后说分拣口令即可提出请求，无需网页按钮。每次口令重新检查现场；忙碌不排队，异常保留停止与复核保护。':'确认现场后开启，再说“小爱同学”和分拣口令。每批结束需重新确认，服务继续在线；120秒未使用或页面失联将取消本批资格。');
 const batch=d.batch_mode===true,unit=batch?'一批':'一件';
 const finalOnly=d.batch_verification_mode==='end_of_batch';
 text('nav-note','固定工位 · '+(finalOnly?'整批末尾核验':'逐件核验')+' · 抓取门槛 '+fmt((d.decision_thresholds?.min_confidence??.7)*100,0)+'%');
 const selected=$('operation-material').value;
 text('operation-start',batch?'开始'+operationMaterials[selected]:'分拣一次');
 for(const button of document.querySelectorAll('button[data-operation-material]')){button.disabled=d.busy||operationRequest||operationPending;button.setAttribute('aria-pressed',String(button.dataset.operationMaterial===selected))}
 text('operation-scope',finalOnly?'每批最多 '+d.max_objects+' 件，抓取门槛 '+fmt((d.decision_thresholds?.min_confidence??.4)*100,0)+'%。完整抓放退回后，本件视觉失败记账并继续下一件，最后统一核验；动作不确定、定位异常或停止请求仍会中止。落点保持原位置，已有物体占用时停止。':d.control_version>=3?'全画面检测，只处理所选类别；其他物品参与避让，箱内物品排除。每批最多 '+d.max_objects+' 件目标，逐件复核，落点使用原位置；落点占用或异常停止。待命先空区、最长等待 120 秒，页面失联请求停止。停止按钮不是硬件急停。':batch?'一批最多 '+d.max_objects+' 件，逐件识别、抓放并重新核对；落点占用或异常立即停下。全位置和全角度仍需实抓验证。待命先确认空区，最长 120 秒等待一批；页面失联会请求停止。停止按钮不是硬件急停。':'待命最长 120 秒：先确认取物区为空，再放入一件，检查通过后执行一次并退出。页面失联会请求停止。停止按钮不是硬件急停。');
 const blocked=!d.configured||d.blockers?.length,own=d.owner===operationOwner;
 if(d.skip_per_item_visual_verification)text('operation-scope','每件完整抓放退回后直接进入下一件的新场景识别，取消逐件视觉核验等待，整批最后统一核验。身份、定位、路径、停止和指令完成检查仍逐件执行。');
 if(d.wait_time_scale===.5)$('operation-scope').textContent+=' 当前采图、通信超时及额外停顿均为原值50%；运动时长和完整回显等待保留。';
 $('operation-scope').textContent=$('operation-scope').textContent.replaceAll('120 秒',String(d.standby_seconds??120)+' 秒');
 $('operation-material').disabled=d.busy||operationRequest;
 $('operation-ready').disabled=d.busy||operationRequest;
 $('operation-preview').disabled=d.busy||operationRequest||!d.configured;
 for(const id of ['operation-start','operation-arm'])$(id).disabled=d.busy||operationRequest||operationPending||blocked||!$('operation-ready').checked;
 $('operation-voice-arm').disabled=!d.voice_service?.connected||d.busy||operationRequest||operationPending||blocked||!$('operation-ready').checked;
 $('operation-cancel').disabled=!own||!['checking','waiting_clear','standby','voice_waiting'].includes(d.state)||operationRequest;
 $('operation-stop').disabled=!d.configured||operationRequest;
 $('operation-reset').disabled=!d.reset_supported||d.busy||operationRequest;
 $('operation-home').disabled=!d.home_supported||d.busy||operationRequest||operationPending||blocked;
 const stopLatched=d.blockers?.includes('operator_stop_requested');
 text('operation-home',stopLatched?'回中位（先重置状态）':'回中位');
 $('operation-home').title=stopLatched?blockerNames.operator_stop_requested:'核对现场后，单独执行低速回中位';
 if(typeof d.empty_bin_max_changed_fraction==='number')$('operation-scope').textContent+=' 空箱外观差异上限 '+fmt(d.empty_bin_max_changed_fraction*100,0)+'%，箱内检测到物体仍视为占用。';
 if(d.min_fixed_markers)$('operation-scope').textContent+=' 固定标记至少 '+d.min_fixed_markers+' 枚，保留位移与图块一致性检查。';
 if(typeof d.initial_marker_shift_px==='number')$('operation-scope').textContent+=' 初始定位位移上限 '+fmt(d.initial_marker_shift_px,1)+' 像素。';
 if(d.review_tolerance_factor>1)$('operation-scope').textContent+=' 当前视觉审核容差 '+fmt(d.review_tolerance_factor,1)+' 倍，工作区坐标稳定上限 '+fmt(d.review_tolerance_factor,1)+' 毫米；等待时间保持原值。';
 if(d.bin_edge_overlap_limit>0)$('operation-scope').textContent+=' 箱框边缘重叠审核上限 '+fmt(d.bin_edge_overlap_limit*100,0)+'%，仅限中心在箱外且内侧审核区无物体；箱内占用仍会停止。';
 let message=d.blockers?.length?'暂不可执行：'+d.blockers.map(x=>blockerNames[x]||'工位资格检查未通过（'+x+'）').join('；'):'';
 if(d.busy)message=d.state==='executing'?(batch?'本批逐件分拣中；每件独立核验。':'单件执行中；结果由原执行器记录。'):d.state==='waiting_clear'?(selected==='AUTO'?'请先移走待抓物品，确认空区后再放入'+unit+'。':'请先移走当前类别的待抓物品，其他类别可留在画面中，再放入新的同类目标。'):d.state==='standby'?'等待放入'+unit+'已测样件，物品分开摆放，放下后立即撤手。剩余 '+d.remaining_s+' 秒。':operationNames[d.state];
 if(d.busy&&d.state==='executing'&&d.active_action==='home')message='正在检查并低速回中位，没有执行分类抓取。';
 if(d.busy&&d.state==='executing'&&d.active_action==='start')message='请求已收到，正在处理本批：先完成定位、识别、路径及空箱检查，再连接机械臂并核对后开始运动。开始前会有等待，请勿重复点击。'+(finalOnly?'每件完整抓放退回后继续，最后统一视觉核验。':'');
 if(d.reason)message=operationReasons[d.reason]||'本次入口已退出，请查看任务记录。';
 if(d.state==='voice_waiting')message='本批已授权：说“小爱同学”后说分拣口令。剩余 '+d.remaining_s+' 秒；结束后需重新确认现场。';
 if(d.result)message=(d.result.preview||d.result.status==='preview_only'?'预演：'+(statuses[d.result.status]||d.result.status)+'，没有发送机械指令。':(statuses[d.result.status]||d.result.status||'执行结果待核对')+'；本次资格已消耗。')+(d.result.task_id?' 任务 '+d.result.task_id:'')+(d.reason?' '+(operationReasons[d.reason]||'已退出。'):'');
 if(d.result?.batch_id)message+=' 本批核验通过 '+(d.result.completed_count||0)+' 件，剩余 '+d.result.remaining_count+' 件。';
 if(finalOnly&&d.result?.batch_id)message+=' 已完成动作 '+(d.result.motion_completed_count||0)+' 件；'+(d.skip_per_item_visual_verification?'逐件核验已延至批末；':'本件视觉失败 '+(d.result.per_item_visual_failures?.length||0)+' 件；')+(d.result.final_verification?.status==='batch_visual_evidence_pass'?'整批最终核验通过。':'整批最终结果待复核。');
 if(d.result?.reason)message+=' '+MonitorMetrics.operationFailure(d.result.reason,d.result.slot_checks||[]);
 if(d.result?.status==='state_reset')message='状态已重置，历史失败及复核记录已保留，没有移动机械臂。请重新核对现场后再开始。';
 if(d.result?.status==='neutral_return_command_completed')message='回中位指令已完整回显并等待结束。请核对实际姿态；本操作没有执行抓取。';
 if(blocked&&(d.result||d.reason))message+=' 暂不可执行：'+d.blockers.map(x=>blockerNames[x]||x).join('；');
 if(operationNotice)message+=' '+operationNotice;
 if(!d.busy&&!blocked&&!operationPending&&!$('operation-ready').checked)message+=' 勾选“已核对现场”后，“开始”和“开启待命”按钮才会亮起。';
 text('operation-message','当前入口：'+operationMaterials[selected]+'。'+(message||'可先预演；核对现场后开始'+unit+'分拣，或开启待命等待放物。'));
 text('page-message',batch?'三类已测样件 · 本批逐件执行；暂停识别不等于停止分拣。':'操控限已核验单件范围；暂停识别不等于停止分拣。');
 renderActivity();
}
$('operation-ready').onchange=()=>{if(operation)renderOperation(operation)};
for(const button of document.querySelectorAll('button[data-operation-material]'))button.onclick=()=>{
 if(operation?.busy||operationRequest||operationPending)return;
 $('operation-material').value=button.dataset.operationMaterial;$('operation-ready').checked=false;
 operationNotice='已切换入口，预演和执行只处理此入口选择的类别。';if(operation)renderOperation(operation);
};
async function operationCommand(action,resetConfirmed=false){
 if(!operation||operationRequest)return;
 operationRequest=true;operationRequestAction=action;operationNotice='';renderOperation(operation);
 const physical=['start','arm','voice_arm','home'].includes(action);
 let acknowledged=false,rejection='';
 try{
  const path=operation.control_version>=3&&['preview','start','arm'].includes(action)?operationPaths[$('operation-material').value]:'/operation';
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:operation.token,action,owner:operationOwner,request_id:crypto.randomUUID(),material:$('operation-material').value,ready:$('operation-ready').checked,reset_confirmed:resetConfirmed,home_confirmed:action==='home'}),signal:AbortSignal.timeout(10000*waitScale())});
  const data=await response.json();acknowledged=true;operationUpdatedAt=Date.now();operationConnected=true;if(!response.ok){rejection=data.error;if(data.operation)renderOperation(data.operation);throw Error('rejected')}
  operationPending=false;if(physical||action==='reset')$('operation-ready').checked=false;renderOperation(data);
 }catch{operationPending=physical&&!acknowledged;const resetReasons={reset_scene_confirmation_required:'请先确认机械臂已停稳、张爪且没有夹物。',reset_lock_owner_unknown:'设备锁归属不明，未强行清除，请核对设备占用。',reset_device_still_owned:'设备仍被进程占用，不能重置；请先停止正在运行的任务。',reset_state_changed_retry_review:'现场任务状态刚发生变化，请重新核对后重置。',operation_busy_or_closed:'任务尚未结束，请先停止并等待结束后再重置。'};operationNotice=resetReasons[rejection]||(acknowledged?'请求被拒绝，请按当前阻断提示处理。':physical?'请求未确认；请查看当前状态和任务记录，不要重复点击。':'操作未确认，请查看当前状态。');if(physical)$('operation-ready').checked=false}
 finally{operationRequest=false;operationRequestAction='';if(operation)renderOperation(operation)}
}
for(const [id,action] of [['preview','preview'],['start','start'],['arm','arm'],['voice-arm','voice_arm'],['cancel','cancel'],['stop','stop']])$('operation-'+id).onclick=()=>operationCommand(action);
$('operation-reset').onclick=()=>{
 const needs=operation?.reset_requires_confirmation;
 if(needs&&!confirm('请现场确认：机械臂已停稳、夹爪张开且没有夹物，人员和手已退出活动区。重置只归档并解除停止/待复核记录，不会回中位；需要回位时再单独点击“回中位”。确认重置？'))return;
 operationCommand('reset',!!needs);
};
$('operation-home').onclick=()=>{
 if(!confirm('这是实际机械动作。请确认：机械臂已停稳、夹爪为空，未断电或手动挪动，人员和手已退出活动区。仅从有完整记录的准备位或中位低速返回，约需5秒运动。确认回中位？'))return;
 operationCommand('home');
};
async function operationPoll(){
 try{
  const d=await json('/operation');operationUpdatedAt=Date.now();operationConnected=true;
  // Only the tab that explicitly started this operation can renew its lease.
  // The independent keepalive continues while this status request is pending.
  if(d.busy||d.operation_id!==operation?.operation_id){operationPending=false;operationNotice=''}
  renderOperation(d);
 }catch{if(operationKeepalive.recent()){operationConnected=true;renderActivity();text('operation-state','控制保活正常 · 状态读取延迟');return;}operationConnected=false;renderActivity();for(const id of ['preview','start','arm','voice-arm','cancel','reset','home'])$('operation-'+id).disabled=true;text('operation-state','控制连接中断');text('operation-message','连接中断会退出待命，执行中会请求停止；请核对现场。')}
 finally{setTimeout(operationPoll,2000*waitScale())}
}
const operationKeepalive=MonitorMetrics.createOperationKeepalive({getOperation:()=>operation,owner:operationOwner,send:body=>json('/operation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),onAck:()=>{operationConnected=true;renderActivity()},onFailure:()=>{operationConnected=false;renderActivity()}});
operationKeepalive.start();window.addEventListener('pagehide',()=>operationKeepalive.stop());
operationPoll();
setInterval(renderActivity,1000);
$('activity-records').onclick=()=>setScreen('tasks');
function badge(id,label,kind=''){$(id).textContent=label;$(id).className='badge '+kind}
function waitScale(){return operation?.wait_time_scale===.5 ? .5 : 1}
async function json(url,options={}){const r=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout((url.startsWith('/operation')?8000:4000)*waitScale()),...options});if(!r.ok)throw Error('request_failed');return r.json()}
function setScreen(next){
 screen=next;for(const key of ['live','tasks','performance','upload']){$('view-'+key).hidden=key!==next;$('tab-'+key).setAttribute('aria-selected',String(key===next));$('tab-'+key).tabIndex=key===next?0:-1}
 if(next==='tasks'&&taskData)renderTask(taskData);
}
for(const key of ['live','tasks','performance','upload']){
 $('tab-'+key).onclick=()=>setScreen(key);
 $('tab-'+key).onkeydown=e=>{const keys=['live','tasks','performance','upload'];let i=keys.indexOf(key);if(e.key==='ArrowRight')i=(i+1)%4;else if(e.key==='ArrowLeft')i=(i+3)%4;else if(e.key==='Home')i=0;else if(e.key==='End')i=3;else return;e.preventDefault();setScreen(keys[i]);$('tab-'+keys[i]).focus()};
}
function setFocus(next){focus=next;$('frames').dataset.focus=next;for(const b of document.querySelectorAll('button[data-focus]'))b.setAttribute('aria-pressed',String(b.dataset.focus===next));if(next==='result')cameraLost('原始画面按需连接')}
for(const b of document.querySelectorAll('button[data-focus]'))b.onclick=()=>setFocus(b.dataset.focus);
function renderRuntimeBaseline(){
 const d=deviceBenchmark;
 if(!MonitorMetrics.baselineMatches(d,health)){
  text('inference-reference','毫秒 · 无匹配对照');text('runtime-speedup','—');text('runtime-reference','暂无与现役模型及精度匹配的 CPU / GPU 实测');
  text('runtime-baseline-scope','实时耗时继续显示；缺少配对对照时不计算加速比。');return;
 }
 text('inference-reference','毫秒 · 已测典型 '+fmt(d.median_speedup,2)+'×');
 text('runtime-speedup',fmt(d.median_speedup,2)+'×');
 text('runtime-reference','CPU FP32 '+fmt(d.cpu_median_ms)+' → GPU '+d.gpu_precision.toUpperCase()+' '+fmt(d.gpu_median_ms)+' ms');
 text('runtime-baseline-scope','典型中位 '+fmt(d.median_speedup,2)+'× · 整批含卡顿 '+fmt(d.batch_speedup,2)+'× · 同为 FP32 '+fmt(d.same_fp32_median_speedup,2)+'×。'+time(d.measured_at)+'，每种设置 '+d.samples_per_profile+' 次；含 FP16 收益。非本帧同步对比。');
}
function clearRuntime(message){
 runtimeWindow.clear();text('live-grasp-ms','—');text('runtime-frame',message);text('runtime-inference','—');text('runtime-grasp','—');
 text('runtime-window','等待新鲜识别计时');text('runtime-grasp-detail','等待本帧候选点计时');
}
function renderRuntime(d){
 const stats=runtimeWindow.add(d,health);
 if(!stats){clearRuntime('等待匹配的新鲜帧');return;}
 text('runtime-frame','相机源帧 '+(d.source_frame_id??'—'));
 text('runtime-inference',fmt(d.inference_ms,1)+' ms');
 text('runtime-window','近 '+stats.samples+' 个网页采样帧 · 中位 '+fmt(stats.median_ms)+' / P95 '+fmt(stats.p95_ms)+' ms');
 const g=MonitorMetrics.geometry(d);
 if(g){text('live-grasp-ms',fmt(g.elapsed_ms,2));text('runtime-grasp',fmt(g.elapsed_ms,2)+' ms');text('runtime-grasp-detail',g.point_count+' 个显示候选点 · 解码 '+fmt(g.mask_decode_ms,2)+' / 选点 '+fmt(g.point_compute_ms,2)+' ms');}
 else{text('live-grasp-ms','—');text('runtime-grasp','—');text('runtime-grasp-detail','当前页面服务尚未提供候选点计时');}
}
$('runtime-open-performance').onclick=()=>setScreen('performance');
function clearLive(message){clearRuntime(message);liveResult=null;lastResultKey=null;resultAt=0;$('result-image').hidden=true;$('result-image').removeAttribute('src');$('candidate-overlay').replaceChildren();$('result-empty').hidden=false;text('result-empty',message);text('frame-caption',message);text('inference-ms','—');for(const k of Object.keys(names))text(k+'-count','—');text('detection-list',message);$('download-live').disabled=true}
function cameraLost(message){$('camera-image').hidden=true;$('camera-empty').hidden=false;text('camera-empty',message);text('camera-caption','独立原始帧 · 等待更新')}
function drawPoints(svg,size,points,planned=false){
 svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+size[0]+' '+size[1]);svg.setAttribute('preserveAspectRatio','xMidYMid meet');
 const ns='http://www.w3.org/2000/svg',color=planned?'#ffc778':'#adf3cf';
 for(const [index,item] of points.entries()){
  const [x,y]=item.pixel;if(!Number.isFinite(x)||!Number.isFinite(y))continue;
  for(const [x1,y1,x2,y2] of [[x-8,y,x+8,y],[x,y-8,x,y+8]]){const line=document.createElementNS(ns,'line');for(const [k,v] of Object.entries({x1,y1,x2,y2,stroke:color,'stroke-width':2}))line.setAttribute(k,v);svg.append(line)}
  const circle=document.createElementNS(ns,'circle');for(const [k,v] of Object.entries({cx:x,cy:y,r:5,fill:'none',stroke:'#fff','stroke-width':1.5}))circle.setAttribute(k,v);svg.append(circle);
  const label=document.createElementNS(ns,'text');for(const [k,v] of Object.entries({x:Math.max(4,Math.min(size[0]-130,x+12)),y:Math.max(16,y-12),fill:color,'font-size':12,stroke:'#07131c','stroke-width':2.5,'paint-order':'stroke'}))label.setAttribute(k,v);label.textContent=(planned?'计划抓点':'候选 '+(index+1))+' ('+x+', '+y+')';svg.append(label);
 }
}
function renderCandidates(){if(liveResult&&$('result-image').complete&&!$('result-image').hidden)drawPoints($('candidate-overlay'),liveResult.image_size||[640,480],$('show-points').checked?liveResult.display_grasp_points||[]:[])}
$('show-points').onchange=renderCandidates;
function detectionList(parent,items){parent.replaceChildren();if(!items.length){parent.textContent='本帧未检出目标';return}for(const item of items){const row=document.createElement('div');row.className='detection-row';const label=document.createElement('span');label.className=Object.hasOwn(names,item.label)?item.label:'';label.textContent=(item.official_label||'未提供细分类')+' · '+(names[item.label]||item.label);const score=document.createElement('span');score.textContent=fmt(item.confidence*100)+'%';row.append(label,score);parent.append(row)}}
async function resultPoll(){
 try{if(health?.live_enabled){const d=await json('/live-result');if(!health.live_enabled)return;if(!Number.isFinite(d.result_age_ms)||d.result_age_ms>=2000||d.result_age_ms<0||d.model_version!==health.model_sha256||d.runtime_id!==health.runtime_id)throw Error('stale_result');resultAt=Date.now()-d.result_age_ms;
  const key=d.model_version+':'+d.source_stream_id+':'+d.frame_id;
  if(key!==lastResultKey){lastResultKey=key;liveResult=d;$('candidate-overlay').replaceChildren();$('result-image').onload=()=>{if(liveResult===d)renderCandidates()};$('result-image').src='data:image/jpeg;base64,'+d.overlay_base64;$('result-image').hidden=false;$('result-empty').hidden=true;for(const k of Object.keys(names))text(k+'-count',d.instances.filter(x=>x.label===k).length);detectionList($('detection-list'),d.instances);$('download-live').disabled=false}
  renderRuntime(d);text('inference-ms',fmt(d.inference_ms,0));text('frame-caption','相机源帧 '+(d.source_frame_id??'—')+' · 识别结果 '+d.frame_id+' · 接收至结果 '+fmt(d.result_age_ms,0)+' ms');
 }}catch{if(health?.live_enabled)clearLive('等待新鲜识别结果')}finally{setTimeout(resultPoll,screen==='live'?220:1200)}
}
async function cameraPoll(){
 try{if(screen==='live'&&focus!=='result'){
  const r=await fetch('/camera.jpg',{cache:'no-store',signal:AbortSignal.timeout(2200*waitScale())});if(!r.ok)throw Error();const age=Number(r.headers.get('X-Frame-Age-Ms'));if(!Number.isFinite(age)||age>=2000||age<0)throw Error();const blob=await r.blob();const old=cameraUrl;cameraUrl=URL.createObjectURL(blob);cameraAt=Date.now()-age;$('camera-image').src=cameraUrl;$('camera-image').hidden=false;$('camera-empty').hidden=true;if(old)URL.revokeObjectURL(old);text('camera-caption','独立相机帧 '+r.headers.get('X-Frame-Id')+' · 帧龄 '+fmt(age,0)+' ms');
 }}catch{cameraLost('相机连接暂不可用')}finally{setTimeout(cameraPoll,screen==='live'&&focus!=='result'?250:1000)}
}
async function healthPoll(){
 try{const d=await json('/health');const identity=MonitorMetrics.identity(d);if(modelIdentity!==identity){modelIdentity=identity;clearLive('模型身份已更新，等待新帧');$('gpu-benchmark').hidden=true;deviceBenchmark=null;health=d;renderRuntimeBaseline();loadPerformance()}health=d;
 text('model-name',d.model_name||'—');text('model-device',d.execution?.device_name?.includes('890M')?'Radeon 890M · '+d.execution.precision.toUpperCase():d.execution?.precision?.toUpperCase()||'');
 badge('camera-state',d.camera_connected?'相机在线':'相机断开',d.camera_connected?'ok':'warn');
 const gpu=d.model_loaded&&d.execution?.device==='cuda:0'&&!!d.execution?.hip;badge('gpu-state',gpu?'ROCm GPU · '+d.execution.precision.toUpperCase():'识别设备待核对',gpu?'ok':'warn');
 $('toggle').disabled=!d.live_available;$('toggle').textContent=d.live_enabled?'暂停识别':'恢复识别';text('fps',d.live_enabled?fmt(d.recognition_fps):'0.0');text('display-threshold','画面阈值 '+fmt(d.confidence_threshold*100,0)+'%');
 if(!d.live_enabled)clearLive('识别已暂停，相机可继续查看');else if(d.inference_error)clearLive('识别暂不可用');
 if(!d.camera_connected)clearLive('相机连接中断，等待恢复');
 updateIdentity();
 }catch{badge('camera-state','相机状态未知','warn');badge('gpu-state','识别连接中断','warn');deviceBenchmark=null;renderRuntimeBaseline();text('fps','—');clearLive('识别服务连接暂不可用')}finally{setTimeout(healthPoll,2000)}
}
$('toggle').onclick=async()=>{if(!health)return;$('toggle').disabled=true;try{await json('/live',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:!health.live_enabled,prompt_mode:health.default_prompt_mode||'supervised'})});health.live_enabled=!health.live_enabled;$('toggle').textContent=health.live_enabled?'暂停识别':'恢复识别';clearLive(health.live_enabled?'等待新鲜识别结果':'识别已暂停，相机可继续查看')}catch{text('frame-caption','识别开关未生效，请稍后重试')}finally{$('toggle').disabled=false}};
setInterval(()=>{if(resultAt&&Date.now()-resultAt>=2000)clearLive('等待新鲜识别结果');if(cameraAt&&Date.now()-cameraAt>=2200)cameraLost('等待新鲜相机帧');if(boardAt&&Date.now()-boardAt>=2000)clearBoard('等待新鲜 FPGA 统计帧')},300);
function renderBoardBenchmark(){
 const b=boardBenchmark;
 if(!MonitorMetrics.boardBenchmarkMatches(b,station)){
  text('fpga-live-speedup','—');text('fpga-live-reference','暂无与当前配对资产匹配的实测对照');
  text('pl-reference-hint','毫秒 · 含整理与读回');
  text('fpga-live-scope','实时处理耗时继续显示；没有匹配对照时不计算加速比。');
  text('fpga-live-pc','电脑 CPU 是另一对照对象，不能与板端 ARM 混用。');return;
 }
 const ratio=b.cpu_median_ms/b.pl_median_ms;
 text('fpga-live-speedup',fmt(ratio,2)+'×');
 text('pl-reference-hint','毫秒 · 板端 ARM 对照已测 '+fmt(ratio,2)+'×');
 text('fpga-live-reference','板端 ARM '+fmt(b.cpu_median_ms)+' → FPGA '+fmt(b.pl_median_ms)+' ms');
 text('fpga-live-scope',b.samples+' 次同输入整数结果一致 · 耗时减少 '+fmt((1-b.pl_median_ms/b.cpu_median_ms)*100)+'%。原部署六区合成输入启动实测，非本帧同步 CPU 对比'+(station?.runtime_regions_observed?'；当前六区随任务重投影。':'。'));
 const pc=b.pc_reference;
 text('fpga-live-pc',typeof pc?.cpu_median_ms==='number'&&Number.isFinite(pc.cpu_median_ms)&&pc.cpu_median_ms>0?'同输入电脑 HX370 CPU 参考 '+fmt(pc.cpu_median_ms)+' ms；上述加速比针对板端 ARM，不代表比电脑 CPU 快或整机分拣加速。':'上述加速比针对板端 ARM；未测整机分拣加速。');
}
function renderBoardRuntime(d){
 const stats=boardWindow.add(d);if(!stats)return false;
 const e=d.evidence;boardAt=Date.now()-d.frame_age_ms;
 text('fpga-live-ms',fmt(e.processing_ms,1)+' ms');
 text('fpga-live-frame','统计帧 '+e.frame_id+' · 基线 '+e.baseline_frame_id+' · 帧龄 '+fmt(d.frame_age_ms,0)+' ms');
 text('fpga-live-window',stats.samples+' 帧 · 中位 '+fmt(stats.median_ms)+' / P95 '+fmt(stats.p95_ms)+' ms');
 badge('fpga-live-state','实时更新','ok');return true;
}
$('fpga-open-performance').onclick=()=>setScreen('performance');
function clearBoard(message){boardAt=0;boardWindow.clear();text('fpga-live-ms','—');text('fpga-live-window','—');text('fpga-live-frame',message);badge('fpga-live-state',message,'warn');text('board-state',message);badge('pl-state',message,'warn');text('pl-ms','—');$('board-content').hidden=true;$('board-unavailable').hidden=false;text('board-unavailable',message);$('region-rows').replaceChildren()}
async function stationPoll(){
 try{const d=await json('/station-vision');station=d;renderBoardBenchmark();
  if(!d.available||d.faulted){clearBoard(d.faulted?'FPGA 处理故障':d.configured&&d.camera_connected?'等待任务基线':'FPGA 连接待恢复');return}
  if(!renderBoardRuntime(d)){clearBoard('等待身份匹配的新鲜统计帧');return;}
  const e=d.evidence;badge('pl-state','FPGA 统计在线','ok');text('board-state',e.consecutive_frames+'/'+d.window_frames+' 帧窗口');text('pl-ms',fmt(e.processing_ms,1));$('board-content').hidden=false;$('board-unavailable').hidden=true;
  const body=$('region-rows');body.replaceChildren();for(const [name,row] of Object.entries(e.regions)){const tr=document.createElement('tr');tr.title='区域面积 '+row.area_px+' px';for(const value of [regions[name]||name,row.changed_px,row.persistent_px]){const td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}
  text('board-frame','统计帧 '+e.frame_id+' · 基线 '+e.baseline_frame_id+' · 帧龄 '+fmt(d.frame_age_ms,0)+' ms');updateIdentity();
 }catch{station=null;renderBoardBenchmark();clearBoard('FPGA 连接待恢复')}finally{setTimeout(stationPoll,1200)}
}
function updateIdentity(){const box=$('runtime-identity');box.replaceChildren();for(const [label,value] of [['模型 SHA256',health?.model_sha256],['识别运行身份',health?.runtime_id],['板端配置 SHA256',station?.config_sha256],['配对 bit SHA256',station?.assets?.['station_vision.bit']],['配对 hwh SHA256',station?.assets?.['station_vision.hwh']]]){const p=document.createElement('p');p.textContent=label+'：'+(value||'当前不可用');box.append(p)}}
async function loadPerformance(){
 let gpu=false,pl=false;const requestedIdentity=MonitorMetrics.identity(health);
 try{const d=await json('/benchmark');if(requestedIdentity===MonitorMetrics.identity(health)&&MonitorMetrics.baselineMatches(d,health)){deviceBenchmark=d;gpu=true;$('gpu-benchmark').hidden=false;text('gpu-ratio',fmt(d.median_speedup,2)+'×');text('gpu-cpu-time',fmt(d.cpu_median_ms)+' ms');text('gpu-gpu-time',fmt(d.gpu_median_ms)+' ms');text('gpu-batch','整批含卡顿 '+fmt(d.batch_speedup,2)+'× · 平均每图 '+fmt(d.cpu_mean_ms)+' → '+fmt(d.gpu_mean_ms)+' ms');text('gpu-conditions',d.model_name+' · '+time(d.measured_at)+' · CPU '+d.cpu_threads+' 线程 · batch '+d.batch+' · 每种设置 '+d.samples_per_profile+' 次');text('gpu-spikes','GPU 最长单帧 '+fmt(d.gpu_max_ms)+' ms；同 FP32 中位速度比 '+fmt(d.same_fp32_median_speedup,2)+'×。'+(d.concurrent_training?'测量时有训练并行。':'')+'FP16 典型收益包含精度优化。')}}catch{}if(requestedIdentity!==MonitorMetrics.identity(health))return;if(!gpu){deviceBenchmark=null;$('gpu-benchmark').hidden=true;}renderRuntimeBaseline();
 try{const d=await json('/station-benchmark');if(MonitorMetrics.boardBenchmarkMatches(d,station)){boardBenchmark=d;pl=true;$('fpga-benchmark').hidden=false;text('fpga-ratio',fmt(d.median_speedup,2)+'×');text('fpga-cpu-time',fmt(d.cpu_median_ms)+' ms');text('fpga-pl-time',fmt(d.pl_median_ms)+' ms');text('fpga-reduction','板端单次耗时减少 '+fmt(d.latency_reduction_percent)+'% · 已保存实测');text('fpga-conditions',d.samples+' 次相同输入的整数结果一致；报告、原部署配置及配对资产指纹已匹配。');text('fpga-pc-reference',d.pc_reference?'同输入本机 HX370 参考 '+fmt(d.pc_reference.cpu_median_ms)+' ms；FPGA 的此项优势针对板端 ARM。':'本机 HX370 不属于上述倍数的对照对象。')}}catch{}if(!pl){boardBenchmark=null;$('fpga-benchmark').hidden=true;}renderBoardBenchmark();$('benchmark-empty').hidden=gpu||pl;
}
async function workflowPoll(){try{workflow=await json('/workflow');const threshold=operation?.decision_thresholds?.min_confidence??workflow.min_confidence;const finalOnly=operation?.batch_verification_mode==='end_of_batch';text('workflow-threshold',workflow.available?'抓取/终检置信度门槛 '+fmt(threshold*100,0)+'% · '+(finalOnly?'整批末尾核验':operation?.batch_mode?'逐件分拣':workflow.motion_profile==='compact'?'快档':'已测路径'):'闭环配置暂不可读');text('nav-note',workflow.available?'固定工位 · '+(finalOnly?'整批末尾核验':operation?.batch_mode?'逐件分拣':'单件')+' · 抓取门槛 '+fmt(threshold*100,0)+'%':'固定工位 · 分拣')}catch{workflow=null;text('workflow-threshold','闭环配置暂不可读')}finally{setTimeout(workflowPoll,10000)}}
function stepName(s){if(stepNames[s])return stepNames[s];if(/^transfer_\d+$/.test(s))return '转运 '+(Number(s.split('_')[1])+1);if(/^retreat_\d+$/.test(s))return '退离 '+(Number(s.split('_')[1])+1);return s}
function outcome(d){return statuses[d.raw_status]||(d.finished?'已结束记录':d.raw_status||'等待记录')}
function updateLatest(d){$('latest-task').disabled=!d.available;if(!d.available){text('latest-task-title',d.reason==='record_write_in_progress'?'日志正在更新':'暂无现役任务记录');text('latest-task-detail','');return}text('latest-task-title',(names[d.label]||'单件任务')+' · '+outcome(d));text('latest-task-detail',fmt(d.elapsed_s,1)+' 秒 · '+d.echoed_commands+'/'+d.planned_commands+' 回显'+(d.operator_review?' · 有人工复核':''))}
$('latest-task').onclick=()=>{taskChoice='';$('task-select').value='';taskSignature=null;recordKey=null;setScreen('tasks');taskData=null;text('task-empty','正在读取最新任务…');$('task-empty').hidden=false;$('task-content').hidden=true};
$('task-select').onchange=()=>{taskChoice=$('task-select').value;taskSignature=null;recordKey=null;taskData=null;$('task-content').hidden=true;$('task-empty').hidden=false;text('task-empty','正在读取所选任务…')};
function taskOptions(records){const select=$('task-select');const signature=JSON.stringify(records);if(select.dataset.signature===signature)return;select.dataset.signature=signature;select.replaceChildren();const first=document.createElement('option');first.value='';first.textContent='自动跟随最新任务';select.append(first);for(const r of records){const o=document.createElement('option');o.value=r.task_id;o.textContent=(names[r.label]||'单件')+' · '+(statuses[r.raw_status]||r.raw_status)+' · '+new Date(r.recorded_at).toLocaleTimeString('zh-CN',{hour12:false});select.append(o)}select.value=taskChoice}
async function taskPoll(){
 try{const chosen=taskChoice;const latest=await json('/control-status');updateLatest(latest);taskOptions(latest.records||[]);text('lease-state',latest.device_lease_present?'设备占用锁存在 · 以执行器为准':'当前无机械设备占用锁');const d=chosen?await json('/control-status?task_id='+encodeURIComponent(chosen)):latest;if(chosen!==taskChoice)return;taskData=d;if(screen==='tasks')renderTask(d)}catch{taskData=null;taskSignature=null;$('task-content').hidden=true;$('task-empty').hidden=false;text('task-empty','任务日志连接暂不可用');text('latest-task-title','任务日志连接暂不可用');text('latest-task-detail','');text('lease-state','设备占用状态未知')}finally{setTimeout(taskPoll,2200)}
}
function renderTask(d){
 if(!d.available){$('task-content').hidden=true;$('task-empty').hidden=false;text('task-empty',d.reason==='record_write_in_progress'?'日志正在更新，等待完整快照。':'暂无现役新视觉链路的完整任务记录。');return}
 $('task-content').hidden=false;$('task-empty').hidden=true;const {record_age_s,...stable}=d,signature=JSON.stringify(stable);if(signature===taskSignature)return;taskSignature=signature;
 const physical=d.record_kind==='physical';text('task-kind',physical?'实物任务记录':d.record_kind==='simulation'?'模拟记录':'预演记录');text('task-title',(names[d.label]||'单件')+' → '+({bin_plastic:'塑料箱',bin_metal:'金属箱',bin_paper:'纸类箱'}[d.bin_id]||'目标未记录'));
 text('task-meta',time(d.recorded_at)+' · '+(d.motion_profile==='compact'?'快档':'已测路径')+' · '+d.task_id+(d.model_matches_monitor===false?' · 与当前模型不同':''));text('task-time',fmt(d.elapsed_s));text('task-echoes',d.echoed_commands+'/'+d.planned_commands);text('task-waits',d.waited_commands+'/'+d.planned_commands);
 text('task-destination',(names[d.label]||'—')+' / '+({bin_plastic:'塑料箱',bin_metal:'金属箱',bin_paper:'纸类箱'}[d.bin_id]||'—'));text('task-pick',xy(d.pick_pixel));text('task-grip',xy(d.grip_xyz_mm||d.grip_xy_mm));const g=d.geometry;text('geometry-source',g?.backend==='pc_numpy'?'电脑 CPU / NumPy':'来源未确认');text('task-area',g?.area_px);text('task-centroid',xy(g?.centroid_px));text('task-axis',g?.axis_deg==null?'—':fmt(g.axis_deg,2)+'°');text('task-compute','抓点与掩码几何：电脑 CPU；RGB 六区统计：板端 FPGA。');
 text('task-outcome',outcome(d));const reasons={no_unique_new_target_in_correct_bin:'正确箱中未得到满足当时阈值的唯一新增目标',post_retreat_frame_expired:'终态图像证据过期'};const reason=(d.reason||'').replace(/^\w+:/,'');text('task-reason',d.reason?(reasons[reason]||reason)+'。原始自动结果保留。':d.visual_verification?.state==='passed'?'保存的视觉核验通过；不表示已完成无人值守或生产质量验收。':'结果只按任务原始记录展示。');
 $('task-board-result').hidden=!d.board_verification;if(d.board_verification)text('task-board-result','FPGA RGB 核验记录：'+(d.board_verification.status==='board_rgb_evidence_pass'?'通过':d.board_verification.status)+' · '+d.board_verification.frames+' 帧');
 $('task-review').hidden=!d.operator_review;if(d.operator_review)text('task-review-text',d.operator_review.text);
 $('task-supplement').hidden=!d.supplemental?.length;if(d.supplemental?.length){const s=d.supplemental[0];text('task-supplement','后续只读补证：'+(s.status==='supplemental_visual_evidence_pass'?'视觉证据通过':s.status)+' · '+s.frames+' 帧 · 门槛 '+fmt(s.min_confidence*100,0)+'%。没有重发动作，原始自动结果未改写。')}
 const body=$('command-rows');body.replaceChildren();for(const item of d.commands){const tr=document.createElement('tr'),step=document.createElement('td'),planned=document.createElement('td'),actual=document.createElement('td'),duration=document.createElement('td');step.textContent=item.index+'. '+stepName(item.step);const code=document.createElement('code');code.textContent=item.planned_command||'无计划指令';planned.append(code);actual.className='state-'+item.state;const label=document.createElement('span');label.textContent=(physical?'':'模拟 / 预演 · ')+(stateNames[item.state]||item.state);actual.append(label);if(item.command){const detail=document.createElement('details'),summary=document.createElement('summary'),sent=document.createElement('code');summary.textContent='实际发送指令';sent.textContent=item.command;detail.append(summary,sent);actual.append(detail)}duration.textContent=fmt(item.duration_ms/1000)+' 秒';const wait=document.createElement('small');wait.textContent=item.completed_wait?'已完成等待':'未记录等待完成';duration.append(wait);tr.append(step,planned,actual,duration);body.append(tr)}
 loadRecord(d);
}
async function loadRecord(d){
 const key=d.task_id+':'+d.grasp_source_frame+':'+JSON.stringify(d.pick_pixel);if(recordKey===key)return;recordKey=key;$('record-image').hidden=true;$('record-point').replaceChildren();$('record-empty').hidden=false;text('record-empty','正在核对原图身份…');text('record-caption','任务原图与计划抓点，独立于实时画面。');
 if(!d.grasp_image_available||!d.pick_pixel){text('record-empty','此任务没有可核对的抓点原图');return}
 try{const r=await json('/grasp-record?task_id='+encodeURIComponent(d.task_id),{signal:AbortSignal.timeout(12000*waitScale())});if(recordKey!==key)return;if(!r.available)throw Error();$('record-image').onload=()=>{if(recordKey!==key)return;$('record-image').hidden=false;$('record-empty').hidden=true;drawPoints($('record-point'),r.image_size,[{pixel:r.pixel}],true)};$('record-image').src='data:'+r.image_mime+';base64,'+r.image_base64;text('record-caption','原始帧 '+r.source_frame_id+' · PNG 与解码像素哈希匹配 · 橙色为该任务的计划抓点。')}catch{if(recordKey===key)text('record-empty','未找到哈希匹配的原图，不在其他画面上绘制旧抓点')}
}
function download(data,prefix){if(!data)return;const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=prefix+'-'+data.frame_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('download-live').onclick=()=>download(liveResult,'live-result');$('download-upload').onclick=()=>download(uploadResult,'image-result');
async function identify(file,title=''){
 if(uploadBusy)return;if(!file){text('upload-status','请先选择图片');return}if(file.size>6000000){text('upload-status','请将图片缩小到 6 MB 以内');return}uploadBusy=true;$('upload-run').disabled=true;$('download-upload').disabled=true;uploadResult=null;$('upload-image').hidden=true;$('upload-empty').hidden=false;text('upload-empty','正在识别图片…');text('upload-status','正在使用现役模型识别'+(title?'：'+title:'')+'…');
 try{const image_base64=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(file)});const d=await json('/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64,prompt_mode:health?.default_prompt_mode||'supervised'}),signal:AbortSignal.timeout(30000*waitScale())});uploadResult=d;$('upload-image').src='/overlay.jpg?frame='+encodeURIComponent(d.frame_id);$('upload-image').hidden=false;$('upload-empty').hidden=true;detectionList($('upload-results'),d.instances);text('upload-status','识别完成 · '+fmt(d.inference_ms,0)+' ms'+(title?' · '+title:''));$('download-upload').disabled=false}catch{text('upload-status','图片识别失败，请检查图片和模型连接');text('upload-empty','未获得可显示结果');text('upload-results','暂无结果')}finally{uploadBusy=false;$('upload-run').disabled=false}
}
$('upload-run').onclick=()=>identify($('upload-file').files[0]);
async function examples(){try{const samples=await json('/examples');for(const s of samples){if(typeof s.url!=='string'||!s.url.startsWith('/examples/'))continue;const b=document.createElement('button');b.textContent=s.title;b.onclick=async()=>{try{const r=await fetch(s.url);if(!r.ok)throw Error();await identify(await r.blob(),s.title)}catch{text('upload-status','样图暂不可用')}};$('examples').append(b)}$('example-details').hidden=!$('examples').childElementCount}catch{}}
$('result-image').onerror=()=>clearLive('识别图像暂不可用');$('camera-image').onerror=()=>cameraLost('相机图像暂不可用');$('record-image').onerror=()=>{$('record-image').hidden=true;$('record-empty').hidden=false;text('record-empty','原图无法显示')};
healthPoll();resultPoll();cameraPoll();stationPoll();taskPoll();workflowPoll();examples();
setInterval(loadPerformance,20000);

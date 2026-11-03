'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../../pynq_km1_vision/src/monitor_assets/monitor.js'),'utf8');
const context={module:{exports:{}},globalThis:{}};
vm.runInNewContext(source.split('// Dashboard bindings.')[0],context);
const {SampleWindow,geometry,baselineMatches}=context.module.exports;

const activity=context.module.exports.operationActivity;
const idle=()=>({control_version:3,configured:true,state:'idle',busy:false,blockers:[]});
test('activity distinguishes operator wait, preview, placement, execution and stop without claiming motion feedback',()=>{
 assert.match(activity(idle()).title,/空闲/);
 assert.match(activity(idle(),{ready:true}).title,/现场已确认/);
 for(const [state,label] of [['checking','检查待命'],['checking_preview','预演'],['waiting_clear','清空'],['standby','等待放入'],['voice_waiting','等待口令'],['executing','分拣处理中'],['stopping','等待执行器']]){
  const a=activity({...idle(),state,busy:true,batch_mode:true,remaining_s:42});
  assert.match(a.title,new RegExp(label));
 }
 assert.match(activity({...idle(),state:'standby',busy:true,remaining_s:42}).detail,/42 秒/);
 const executing=activity({...idle(),state:'executing',busy:true});
 assert.match(executing.detail,/未报告具体机械步骤/);
 assert.doesNotMatch(executing.title,/正在抓取|正在转运|已经停稳/);
 assert.match(activity({...idle(),state:'executing',busy:true,active_action:'home'}).title,/回中位/);
});
test('activity prioritizes lost connection, unknown requests and protected state over old successful results',()=>{
 const success={...idle(),state:'finished',result:{status:'completed',final_verification:{status:'batch_visual_evidence_pass'}}};
 assert.equal(activity(success).tone,'success');
 assert.match(activity(success,{connected:false}).title,/状态未知/);
 assert.match(activity(success,{pending:true}).title,/未确认/);
 assert.match(activity(success,{requestAction:'start'}).title,/提交分拣/);
 assert.equal(activity({...success,blockers:['operator_stop_requested']}).tone,'error');
 assert.match(activity({...success,blockers:['operator_stop_requested']}).title,/停止保护/);
 assert.equal(activity({...success,configured:false}).tone,'error');
 assert.equal(activity({...success,control_version:99}).tone,'error');
 assert.equal(activity({...idle(),state:'unexpected'}).tone,'error');
});
test('activity reports failures, preview and reset separately from visual success',()=>{
 assert.equal(activity({...idle(),state:'rejected',result:{status:'rejected',reason:'marker_reprojection_error'}}).tone,'error');
 assert.match(activity({...idle(),state:'rejected',result:{reason:'marker_reprojection_error'}}).detail,/定位误差/);
 assert.match(activity({...idle(),state:'finished',result:{status:'preview_only'}}).title,/预演已结束/);
 assert.equal(activity({...idle(),state:'finished',result:{status:'selection_complete'}}).tone,'waiting');
 assert.equal(activity({...idle(),state:'finished',result:{status:'completed'}}).tone,'waiting');
 assert.match(activity({...idle(),result:{status:'state_reset'}}).detail,/没有移动机械臂/);
 assert.match(activity({...idle(),result:{status:'neutral_return_command_completed'}}).detail,/核对实际姿态/);
});
test('operation failures explain contour rejection and file conflicts without exposing paths',()=>{
 const explain=context.module.exports.operationFailure;
 assert.match(explain("ValueError:unified_no_pickable_instance:[{'reason':'unified_fragmented_specimen'}]"),/轮廓分成多块/);
 assert.match(explain('PermissionError:[WinError 32] C:\\private\\task.json'),/文件暂被占用/);
 assert.doesNotMatch(explain('PermissionError:[WinError 32] C:\\private\\task.json'),/private/);
 assert.equal(explain(null),'');
 assert.match(explain('ValueError:motion_requires_two_fixed_markers'),/不足2枚/);
 assert.match(explain('ValueError:marker_reprojection_error'),/定位误差/);
 assert.match(explain('unified_no_free_verified_bin_slot:metal',[{label:'metal',reason:'empty_reference_mismatch',
  occupied_px:0,changed_fraction:.07,max_changed_fraction:.05}]),/金属箱.*7.0%.*5.0%/);
 assert.match(explain('unified_no_free_verified_bin_slot:metal',[{label:'metal',reason:'slot_object_detected',occupied_px:4}]),/检测到物体/);
});
test('batch final mode renders forty percent and separate motion/final verdicts using real HTML IDs',()=>{
 const html=fs.readFileSync(path.join(__dirname,'../../pynq_km1_vision/src/monitor_assets/dashboard.html'),'utf8');
 const nodes={};for(const match of html.matchAll(/id="([^"]+)"/g))nodes[match[1]]={textContent:'',checked:false,value:'AUTO',setAttribute(){}};
 const buttons=['AUTO','PAPER','METAL','PLASTIC'].map(value=>({dataset:{operationMaterial:value},setAttribute(){}}));
 const get=id=>{assert.ok(nodes[id],'missing real HTML node '+id);return nodes[id]};
 const sandbox={crypto:{randomUUID:()=> 'page-owner'},document:{querySelectorAll:()=>buttons},$:get,
  text:(id,value)=>get(id).textContent=value,fmt:(x,n)=>Number(x).toFixed(n),MonitorMetrics:context.module.exports,
  statuses:{completed:'整批视觉核验通过'}};
 vm.createContext(sandbox);
 vm.runInContext(source.slice(source.indexOf('let operation=null'),source.indexOf("$('operation-ready').onchange")),sandbox);
 sandbox.data={control_version:3,batch_mode:true,state:'finished',configured:true,busy:false,blockers:[],
  max_objects:6,reset_supported:true,batch_verification_mode:'end_of_batch',decision_thresholds:{min_confidence:.4},
  result:{batch_id:'batch',status:'completed',completed_count:3,motion_completed_count:3,remaining_count:0,
   per_item_visual_failures:[{}],final_verification:{status:'batch_visual_evidence_pass'}}};
 vm.runInContext('renderOperation(data)',sandbox);
 assert.match(nodes['nav-note'].textContent,/40%/);assert.match(nodes['nav-note'].textContent,/整批末尾/);
 assert.match(nodes['operation-message'].textContent,/已完成动作 3 件/);
 assert.match(nodes['operation-message'].textContent,/本件视觉失败 1 件/);
 assert.match(nodes['operation-message'].textContent,/整批最终核验通过/);
 sandbox.data.skip_per_item_visual_verification=true;sandbox.data.wait_time_scale=.5;
 sandbox.data.standby_seconds=60;vm.runInContext('renderOperation(data)',sandbox);
 assert.match(nodes['operation-scope'].textContent,/取消逐件视觉核验等待/);
 assert.match(nodes['operation-scope'].textContent,/50%/);
 assert.match(nodes['operation-message'].textContent,/逐件核验已延至批末/);
 assert.doesNotMatch(nodes['operation-message'].textContent,/本件视觉失败/);
 assert.equal(nodes['operation-reset'].disabled,false);
 sandbox.data.busy=true;vm.runInContext('renderOperation(data)',sandbox);
 assert.equal(nodes['operation-reset'].disabled,true);
});
test('banner ages out stale state and local UI changes cannot renew its freshness',()=>{
 const html=fs.readFileSync(path.join(__dirname,'../../pynq_km1_vision/src/monitor_assets/dashboard.html'),'utf8');
 const nodes={};for(const match of html.matchAll(/id="([^"]+)"/g))nodes[match[1]]={textContent:'',checked:false,value:'AUTO',attrs:{},setAttribute(k,v){this.attrs[k]=v}};
 let now=10000;
 class Clock extends Date{static now(){return now}}
 const sandbox={Date:Clock,crypto:{randomUUID:()=> 'page-owner'},document:{querySelectorAll:()=>[]},$:id=>nodes[id],
  text:(id,value)=>nodes[id].textContent=value,fmt:(x,n)=>Number(x).toFixed(n),MonitorMetrics:context.module.exports,statuses:{}};
 vm.createContext(sandbox);
 vm.runInContext(source.slice(source.indexOf('let operation=null'),source.indexOf("$('operation-ready').onchange")),sandbox);
 sandbox.data=idle();
 vm.runInContext('operationUpdatedAt=Date.now();renderOperation(data)',sandbox);
 assert.match(nodes['activity-title'].textContent,/空闲/);
 now+=6500;
 vm.runInContext('renderOperation(data)',sandbox);
 assert.match(nodes['activity-title'].textContent,/状态未知/);
 assert.equal(nodes['activity-banner'].attrs['data-tone'],'error');
 assert.equal(nodes['operation-preview'].disabled,true);
 vm.runInContext('operationUpdatedAt=Date.now();renderOperation(data)',sandbox);
 assert.match(nodes['activity-title'].textContent,/空闲/);
 assert.equal(nodes['operation-preview'].disabled,false);
 sandbox.data={...idle(),busy:true,state:'executing',operation_id:'new-op'};
 vm.runInContext('renderOperation(data)',sandbox);now+=3000;
 vm.runInContext('operationUpdatedAt=Date.now();renderOperation(data)',sandbox);
 assert.match(nodes['activity-clock'].textContent,/3 秒/);
 sandbox.data.state='stopping';vm.runInContext('renderOperation(data)',sandbox);
 assert.match(nodes['activity-clock'].textContent,/0 秒/);
});
const health=()=>({model_sha256:'model',runtime_id:'runtime',camera_connected:true,live_enabled:true,
 execution:{device:'cuda:0',precision:'fp16',hip:'7.14'}});
function observation(frame=1,ms=40){return {model_version:'model',runtime_id:'runtime',
 source_stream_id:'camera',source_frame_id:String(frame),frame_id:frame,result_age_ms:10,
 inference_ms:ms,instances:[{}],display_grasp_points:[{pixel:[4,5]}]};}
function timed(){const d=observation();d.display_geometry_timing={schema_version:'monitor-display-geometry-timing-v1',
 backend:'pc_cpu_opencv_numpy',scope:'display_candidates_all_instances',elapsed_ms:3,mask_decode_ms:1,
 point_compute_ms:1.5,point_count:1,decoded_masks:1,calibrated:false,control_authorized:false};
 for(const k of ['frame_id','source_frame_id','source_stream_id','model_version','runtime_id'])d.display_geometry_timing[k]=d[k];return d;}
test('repeated polling is counted once, while slow frames remain in the sample window',()=>{
 const w=new SampleWindow(3),h=health();w.add(observation(1,20),h);w.add(observation(1,20),h);
 assert.equal(w.add(observation(2,1000),h).samples,2);
 let s=w.add(observation(3,40),h);assert.equal(s.median_ms,40);assert.ok(s.p95_ms>900);
 s=w.add(observation(4,50),h);assert.equal(s.samples,3);assert.equal(s.median_ms,50);
});
test('stale frames, invalid time, other runtime, pause or camera loss clear live numbers',()=>{
 for(const mutate of [d=>d.result_age_ms=2000,d=>d.result_age_ms=-1,d=>d.inference_ms=NaN,
 d=>d.inference_ms=-1,d=>d.model_version='other',d=>d.runtime_id='other']){
  const w=new SampleWindow();w.add(observation(),health());const d=observation(2);mutate(d);
  assert.equal(w.add(d,health()),null);assert.equal(w.rows.length,0);
 }
 for(const field of ['live_enabled','camera_connected']){const h=health();h[field]=false;assert.equal(new SampleWindow().add(observation(),h),null);}
});
test('a new stream or model runtime starts a separate window',()=>{
 const w=new SampleWindow(),h=health();w.add(observation(1,1000),h);
 let d=observation(2);d.source_stream_id='new';assert.equal(w.add(d,h).samples,1);
 h.runtime_id='new-runtime';d=observation(3,30);d.runtime_id=h.runtime_id;
 assert.equal(w.add(d,h).median_ms,30);assert.equal(w.rows.length,1);
});
test('CPU candidate timing must match all five source identities',()=>{
 assert.equal(geometry(timed()).elapsed_ms,3);
 for(const k of ['frame_id','source_frame_id','source_stream_id','model_version','runtime_id']){
  const d=timed();d.display_geometry_timing[k]='wrong';assert.equal(geometry(d),null);
 }
});
test('CPU timing rejects fabricated GPU, control, counts or invalid duration claims',()=>{
 for(const [key,value] of [['backend','gpu'],['scope','motion_plan'],['control_authorized',true],
 ['point_count',2],['decoded_masks',2],['elapsed_ms',-1],['elapsed_ms',NaN],['elapsed_ms',1]]){
  const d=timed();d.display_geometry_timing[key]=value;assert.equal(geometry(d),null);
 }
 assert.equal(geometry(observation()),null);
});
function baseline(){return {available:true,source:'offline_benchmark',live_speedup:false,
 model_sha256:'model',gpu_precision:'fp16',cpu_median_ms:100,gpu_median_ms:200,
 median_speedup:.5,batch_speedup:.4,same_fp32_median_speedup:.3};}
test('saved comparison preserves slowdowns and never becomes a live speedup claim',()=>{
 assert.equal(baselineMatches(baseline(),health()),true);
 for(const [key,value] of [['model_sha256','other'],['gpu_precision','fp32'],['live_speedup',true],
 ['source','live'],['cpu_median_ms',Infinity],['median_speedup',0]]){
  const b=baseline();b[key]=value;assert.equal(baselineMatches(b,health()),false);
 }
});
const {boardSample,boardBenchmarkMatches,BoardWindow}=context.module.exports;
function board(frame=1,ms=70){return {available:true,configured:true,camera_connected:true,faulted:false,
 contract:'station-vision-v1',backend:'pl_board_local',stream_id:'board-stream',config_sha256:'cfg',
 assets:{'station_vision.bit':'a'.repeat(64),'station_vision.hwh':'b'.repeat(64)},frame_age_ms:100,
 frame_id:frame,baseline_frame_id:0,evidence:{backend:'pl_board_local',config_sha256:'cfg',
 automatic_motion_allowed:false,frame_id:frame,baseline_frame_id:0,processing_ms:ms}};}
function boardBaseline(){return {available:true,source:'saved_board_startup_benchmark',live_speedup:false,
 whole_system_speedup_measured:false,assets:board().assets,config_sha256:'cfg',samples:12,
 cpu_median_ms:165,pl_median_ms:69,median_speedup:165/69};}
test('FPGA live times require fresh frame, matching baseline/configuration and board backend',()=>{
 assert.equal(boardSample(board()).processing_ms,70);
 for(const mutate of [d=>d.frame_age_ms=2000,d=>d.frame_age_ms=-1,d=>d.camera_connected=false,
 d=>d.faulted=true,d=>d.available=false,d=>d.backend='cpu_reference',d=>d.assets={},
 d=>d.evidence.processing_ms=NaN,d=>d.evidence.processing_ms=-1,d=>d.evidence.frame_id=9,
 d=>d.evidence.baseline_frame_id=1,d=>d.evidence.config_sha256='other',d=>d.evidence.automatic_motion_allowed=true]){
  const d=board();mutate(d);assert.equal(boardSample(d),null);
 }
});
test('FPGA sampling deduplicates frames and resets after stream, ROI, baseline or asset changes',()=>{
 const w=new BoardWindow(3);w.add(board(1,65));assert.equal(w.add(board(1,65)).samples,1);
 assert.equal(w.add(board(2,75)).median_ms,70);
 const changed=board(3,80);changed.stream_id='other';assert.equal(w.add(changed).samples,1);
 changed.config_sha256=changed.evidence.config_sha256='new-regions';assert.equal(w.add(changed).samples,1);
 changed.baseline_frame_id=changed.evidence.baseline_frame_id=2;assert.equal(w.add(changed).samples,1);
 changed.assets['station_vision.bit']='c'.repeat(64);assert.equal(w.add(changed).samples,1);
 changed.frame_age_ms=3000;assert.equal(w.add(changed),null);assert.equal(w.rows.length,0);
});
test('FPGA comparison binds paired assets and allows explicit task ROI reprojection only',()=>{
 const b=boardBaseline(),d=board();assert.equal(boardBenchmarkMatches(b,d),true);
 d.config_sha256='task-regions';assert.equal(boardBenchmarkMatches(b,d),false);
 d.runtime_regions_observed=true;assert.equal(boardBenchmarkMatches(b,d),true);
 d.assets['station_vision.bit']='c'.repeat(64);assert.equal(boardBenchmarkMatches(b,d),false);
});
test('saved FPGA comparison rejects live or whole-system speedup claims and preserves slowdowns',()=>{
 for(const [key,value] of [['source','unverified'],['live_speedup',true],['whole_system_speedup_measured',true],
 ['cpu_median_ms',Infinity],['pl_median_ms',0],['samples',0]]){
  const b=boardBaseline();b[key]=value;assert.equal(boardBenchmarkMatches(b,board()),false);
 }
 const b=boardBaseline();b.cpu_median_ms=30;b.pl_median_ms=80;b.median_speedup=30/80;
 assert.equal(boardBenchmarkMatches(b,board()),true);
});


'use strict';
// This browser is a client. PYNQ admits, plans and journals the operation.
window.addEventListener('load',()=>{
 const frame=document.querySelector('iframe[src="/prompted/"]');
 if(!frame)return;
 function bind(){
  const doc=frame.contentDocument,button=doc?.getElementById('plan'),output=doc?.getElementById('planResult');
  if(!button||!output)return;
  button.onclick=async()=>{
   button.disabled=true;let current=null,owner='embedded-'+crypto.randomUUID(),request=crypto.randomUUID();
   async function api(path,body){
    const response=await fetch(path,{cache:'no-store',...(body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});
    const value=await response.json();if(!response.ok)throw new Error(value.error||'板端请求未完成');return value;
   }
   try{
    current=await api('/operation');
    if(current.busy)throw new Error('板端正在处理其他任务，请等待。');
    if(current.board_ownership?.status==='blocked')throw new Error('停止保护已锁存，请先按原流程核对现场。');
    output.textContent='板端正在读取当前单帧并规划；机械臂不会移动。';
    current=await api('/operation',{action:'preview',material:'AUTO',token:current.token,owner,request_id:request});
    const deadline=performance.now()+90000;
    while(current.busy){
     if(current.owner!==owner||current.operation_id!==request)throw new Error('控制会话已变化，请勿重复请求。');
     await api('/operation',{action:'heartbeat',token:current.token,owner,operation_id:request});
     await new Promise(resolve=>setTimeout(resolve,750));
     current=await api('/operation');
     if(performance.now()>deadline)throw new Error('板端预演超时，不自动重试。');
    }
    if(current.operation_id!==request)throw new Error('控制会话已变化，请勿重复请求。');
    const report=await api('/board-preview-report?operation_id='+encodeURIComponent(request));
    const lines=[report.status==='preview_ready'?'板端已生成路线预览，机械臂未移动。':'当前没有可用抓取路线。'];
    if(report.reason)lines.push(report.reason);
    for(const item of report.candidates)lines.push(item.choice.official_label+'；高度 '+item.choice.object_height_mm+' mm；'+item.steps.length+' 个运动步骤。');
    for(const item of report.rejected)lines.push((item.official_label||item.label||'候选')+'：'+item.reason);
    lines.push('规划与任务记录在 PYNQ；新类别和高度仍需实物验证。');output.textContent=lines.join('\n');
   }catch(error){
    output.textContent=error.message;
    if(current?.busy&&current.owner===owner&&current.operation_id===request){
     try{await api('/operation',{action:'stop',token:current.token,owner});}catch{}
    }
   }finally{button.disabled=false;}
  };
 }
 frame.addEventListener('load',bind);bind();
});

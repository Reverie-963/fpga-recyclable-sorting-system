const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const text=fs.readFileSync(path.join(__dirname,'../../pynq_km1_vision/src/monitor_assets/monitor.js'),'utf8');
const context={module:{exports:{}},setTimeout,clearTimeout,Date};
vm.runInNewContext(text.split('// Dashboard bindings.')[0],context);
const {createOperationKeepalive}=context.module.exports;
(async()=>{
 let op={busy:true,owner:'page',operation_id:'one',token:'private'},queue=[],calls=[],now=1000;
 const keep=createOperationKeepalive({getOperation:()=>op,owner:'page',clock:()=>now,
  schedule:fn=>{queue.push(fn);return queue.length},cancel:()=>{},
  send:async body=>{calls.push(body);return {heartbeat_accepted:true,operation_id:body.operation_id}}});
 // No status-poll completion is supplied: the independent client must still renew.
 keep.start();await new Promise(setImmediate);assert.equal(calls.length,1);assert.equal(calls[0].action,'heartbeat');
 now+=2000;await queue.shift()();assert.equal(calls.length,2);assert.equal(calls[1].operation_id,'one');
 op={...op,owner:'another-page'};await queue.shift()();assert.equal(calls.length,2);
 op={...op,owner:'page',busy:false};await queue.shift()();assert.equal(calls.length,2);
 keep.stop();assert.equal(calls.some(c=>c.action==='start'),false);
 console.log('PASS: independent real-client heartbeat, owner/operation matching, no idle/start requests');
})().catch(e=>{console.error(e);process.exitCode=1});

'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../../pynq_km1_vision/src/monitor_assets/model_switch.js'),'utf8');
const {groupModels,initialTab}=require('../../pynq_km1_vision/src/monitor_assets/model_switch.js');
const models=[['yoloe26',26],['v8_3',3],['11s_3',3],['v8_26',26],['11s_26',26],['yoloe24',24],['yoloe26_tuned',26],['yoloe26_reference',26]]
  .map(([id,class_count])=>({id,name:id,class_count,available:true,bound:id==='yoloe26'}));

// A DOM fixture exercises the shipped listeners without a browser, network,
// model loading or hardware. Native tab behavior is represented by clicks.
function fixture({saved=null,unavailable=false}={}) {
  class Element {
    constructor(tag){this.tagName=tag;this.children=[];this.attributes={};this.dataset={};this.listeners={};this.hidden=false;this.disabled=false;this.ownText='';}
    set textContent(value){this.ownText=value;this.children=[];}
    get textContent(){return this.ownText+this.children.map(child=>child.textContent).join('');}
    setAttribute(key,value){this.attributes[key]=String(value);}
    getAttribute(key){return this.attributes[key]??null;}
    append(...nodes){for(const node of nodes){if(node.parent)node.parent.children.splice(node.parent.children.indexOf(node),1);node.parent=this;this.children.push(node);}}
    prepend(node){if(node.parent)node.parent.children.splice(node.parent.children.indexOf(node),1);node.parent=this;this.children.unshift(node);}
    before(node){const i=this.parent.children.indexOf(this);node.parent=this.parent;this.parent.children.splice(i,0,node);}
    replaceChildren(...nodes){for(const node of this.children)node.parent=null;this.children=[];this.append(...nodes);}
    addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
    async click(){if(!this.disabled){if(this.onclick)await this.onclick();for(const fn of this.listeners.click??[])await fn();}}
    showModal(){this.open=true;}
    close(){this.open=false;}
    contains(node){return node===this||this.descendants().includes(node);}
    focus(){document.activeElement=this;}
    async fire(name,event={}){for(const fn of this.listeners[name]??[])await fn(event);}
    descendants(){return this.children.flatMap(child=>[child,...child.descendants()]);}
    querySelector(selector){return this.querySelectorAll(selector)[0]??null;}
    querySelectorAll(selector){return this.descendants().filter(node=>
      selector==='[role="tab"]'?node.getAttribute('role')==='tab':
      selector==='[role="tab"][aria-selected="true"]'?node.getAttribute('role')==='tab'&&node.getAttribute('aria-selected')==='true':
      selector.startsWith('.')?node.className===selector.slice(1):node.tagName===selector);}
  }
  const root=new Element('document'),head=new Element('head'),body=new Element('body');root.append(head,body);
  const nav=new Element('nav');nav.className='view-tabs';body.append(nav);
  const panels={};
  for(const key of ['prompted','live','tasks','performance','upload']) {
    const tab=new Element('button');tab.id='tab-'+key;tab.setAttribute('role','tab');nav.append(tab);
    const panel=new Element('section');panel.id='view-'+key;panels[key]=panel;body.append(panel);
    tab.onclick=()=>{for(const item of nav.children){item.setAttribute('aria-selected',String(item===tab));panels[item.id.slice(4)].hidden=item!==tab;}};
  }
  panels.prompted.append(new Element('p'),new Element('button'),new Element('iframe'));
  const metrics=new Element('section');metrics.className='metrics-strip';
  const primary=new Element('section');primary.className='live-primary';
  const image=new Element('img');image.id='result-image';primary.append(image);
  const detections=new Element('section');detections.className='detection-panel';
  panels.live.append(metrics,primary,detections);
  nav.children[0].onclick(); // Original embedded recognition default.
  const documentListeners={};
  const document={head,body,activeElement:null,createElement:tag=>new Element(tag),
    addEventListener:(name,fn)=>(documentListeners[name]??=[]).push(fn),
    getElementById:id=>root.descendants().find(node=>node.id===id),
    querySelector:selector=>root.querySelector(selector),querySelectorAll:selector=>root.querySelectorAll(selector)};
  const storage=new Map(saved?[['model-switch-active-tab',saved]]:[]);
  const state={selected:'yoloe26',selected_name:'Bound',bound_name:'Bound',motion_model_matches:true,
    revision:0,token:'fixture-page-token',switching:false,models:structuredClone(models),selected_configuration:{confidence:null}};
  if(unavailable){state.models.find(row=>row.id==='11s_26').available=false;state.models.find(row=>row.id==='11s_26').disabled_reason='训练未完成';}
  const calls=[],loads=[],intervals=[];let reloads=0,fail=false;
  const context={document,window:{addEventListener:(name,fn)=>{if(name==='load')loads.push(fn);}},
    sessionStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    location:{reload:()=>{reloads++;}},setInterval:fn=>intervals.push(fn),
    fetch:async(url,options={})=>{calls.push({url,options});if(fail)throw Error('disconnected');return {ok:true,json:async()=>options.method==='POST'?{...structuredClone(state),switching:true}:structuredClone(state)};}};
  vm.runInNewContext(source,context);loads.forEach(fn=>fn());
  return {node:document.getElementById,nav,panels,state,calls,storage,document,
    fire:async(name,event)=>{for(const fn of documentListeners[name]??[])await fn(event);},poll:()=>intervals[0](),
    settle:()=>new Promise(resolve=>setImmediate(resolve)),fail:()=>{fail=true;},recover:()=>{fail=false;},reloads:()=>reloads};
}

test('group by model type before class counts and inference variants',()=>{
  const groups=groupModels(models);
  assert.deepEqual(groups.map(g=>[g.name,g.models.map(m=>m.id)]),[
    ['YOLOE-11m',['yoloe26','yoloe24','yoloe26_tuned','yoloe26_reference']],
    ['YOLOv8s',['v8_3','v8_26']],['YOLO11s',['11s_3','11s_26']],
  ]);
  assert.equal(groupModels([{id:'future',class_count:99}])[0].id,'other');
});
test('fresh console keeps image and actions together; only a valid switch tab restores',async()=>{
  const ui=fixture();await ui.settle();
  assert.equal(ui.nav.children[0].id,'tab-live');
  assert.equal(ui.node('tab-live').textContent,'控制台');
  assert.equal(ui.node('tab-live').getAttribute('aria-selected'),'true');
  assert.equal(ui.panels.prompted.hidden,true);
  assert.equal(ui.panels.live.querySelector('img'),ui.node('result-image'));
  assert.equal(ui.node('recognition-details'),undefined);
  assert.equal(ui.calls.some(c=>c.options.method==='POST'),false);
  assert.equal(initialTab('removed-tab',['tab-live']),'tab-live');
  const restored=fixture({saved:'tab-tasks'});await restored.settle();
  assert.equal(restored.node('tab-tasks').getAttribute('aria-selected'),'true');
  assert.equal(restored.storage.size,0);
});
test('anchored menu has model types and hover shows configurations without a modal or POST',async()=>{
  const ui=fixture();await ui.settle();await ui.node('model-choose').click();
  assert.equal(ui.node('model-menu').hidden,false);
  assert.equal(ui.node('model-choose').getAttribute('aria-expanded'),'true');
  assert.equal(ui.node('model-families').children.length,3);
  assert.equal(ui.node('model-configurations').children.length,4);
  assert.equal(ui.node('model-catalog'),undefined);
  assert.equal(ui.node('model-apply'),undefined);
  await ui.node('model-family-yolov8').fire('pointerenter');
  assert.equal(ui.node('model-config-title').textContent,'YOLOv8s · 配置');
  assert.equal(ui.node('model-configurations').children.length,2);
  assert.equal(ui.calls.some(c=>c.options.method==='POST'),false);
  await ui.node('model-choose').click();assert.equal(ui.node('model-menu').hidden,true);
});
test('clicking a configuration immediately requests that exact ID once and closes menu',async()=>{
  const ui=fixture();await ui.settle();await ui.node('model-choose').click();
  const chosen=ui.node('model-config-yoloe26_reference');
  await chosen.click();await chosen.click();
  const writes=ui.calls.filter(c=>c.options.method==='POST');assert.equal(writes.length,1);
  assert.equal(writes[0].url,'/model-switch/select');
  assert.deepEqual(JSON.parse(writes[0].options.body),{model_id:'yoloe26_reference',revision:0,token:'fixture-page-token'});
  assert.equal(ui.node('model-menu').hidden,true);
  assert.equal(ui.node('model-choose').disabled,true);
});
test('current configuration is checked and clicking it closes without loading',async()=>{
  const ui=fixture();await ui.settle();await ui.node('model-choose').click();
  assert.equal(ui.node('model-config-yoloe26').getAttribute('aria-checked'),'true');
  assert.equal(ui.node('model-config-yoloe26_reference').getAttribute('aria-checked'),'false');
  await ui.node('model-config-yoloe26').click();
  assert.equal(ui.node('model-menu').hidden,true);
  assert.equal(ui.calls.some(c=>c.options.method==='POST'),false);
});
test('polling preserves the hovered model; failed status cannot be bypassed by hover',async()=>{
  const ui=fixture();await ui.settle();await ui.node('model-choose').click();
  await ui.node('model-family-yolov8').fire('pointerenter');
  const item=ui.node('model-config-v8_3');await ui.poll();
  assert.equal(ui.node('model-config-v8_3'),item);
  ui.fail();await ui.poll();await ui.node('model-family-yoloe').fire('pointerenter');
  assert.equal(ui.node('model-config-yoloe26_tuned').disabled,true);
  await ui.node('model-config-yoloe26_tuned').click();
  assert.equal(ui.calls.some(c=>c.options.method==='POST'),false);
  ui.recover();await ui.poll();assert.equal(ui.node('model-config-yoloe26_tuned').disabled,false);
});
test('unavailable configuration stays disabled, including direct click',async()=>{
  const ui=fixture({unavailable:true});await ui.settle();await ui.node('model-choose').click();
  await ui.node('model-family-yolo11').click();
  assert.equal(ui.node('model-config-11s_26').disabled,true);await ui.node('model-config-11s_26').click();
  assert.equal(ui.calls.some(c=>c.options.method==='POST'),false);
});
test('outside click, Escape and arrow keys navigate/close without applying',async()=>{
  const ui=fixture();await ui.settle();await ui.node('model-choose').click();
  const family=ui.node('model-family-yoloe');family.focus();
  const event=key=>({key,preventDefault(){}});
  await ui.node('model-families').fire('keydown',event('ArrowRight'));
  assert.equal(ui.document.activeElement.id,'model-config-yoloe26');
  await ui.node('model-configurations').fire('keydown',event('ArrowDown'));
  assert.equal(ui.document.activeElement.id,'model-config-yoloe24');
  await ui.node('model-configurations').fire('keydown',event('ArrowLeft'));
  assert.equal(ui.document.activeElement,family);
  await ui.fire('keydown',event('Escape'));
  assert.equal(ui.node('model-menu').hidden,true);
  assert.equal(ui.document.activeElement,ui.node('model-choose'));
  await ui.node('model-choose').click();await ui.fire('pointerdown',{target:ui.document.body});
  assert.equal(ui.node('model-menu').hidden,true);
  assert.equal(ui.calls.some(c=>c.options.method==='POST'),false);
});
test('model publication clears old page while retaining console view',async()=>{
  const ui=fixture();await ui.settle();ui.state.revision=1;await ui.poll();
  assert.equal(ui.reloads(),1);assert.equal(ui.storage.get('model-switch-active-tab'),'tab-live');
});

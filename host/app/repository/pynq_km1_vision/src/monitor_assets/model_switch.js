// Publish the inline selection after the configured execution binding commits.
(() => {
  const families=[
    {id:'yoloe',name:'YOLOE-11m',description:'26 类与原始 24 类'},
    {id:'yolov8',name:'YOLOv8s',description:'三类专项 / 公开 26 类'},
    {id:'yolo11',name:'YOLO11s',description:'三类专项 / 公开 26 类'},
    {id:'other',name:'其他型号',description:'其他已登记配置'},
  ];
  function groupModels(models) {
    const groups=families.map(family=>({...family,models:[]}));
    for(const model of models) {
      const id=model.id.startsWith('yoloe')?'yoloe':model.id.startsWith('v8_')?'yolov8':model.id.startsWith('11s_')?'yolo11':'other';
      groups.find(group=>group.id===id).models.push(model);
    }
    return groups.filter(group=>group.models.length);
  }
  function modelDetail(model) {
    if(model.id==='yoloe26')return '现役 26 类 · 原阈值与空背景设置';
    if(model.id==='yoloe26_tuned')return '新版 26 类 · 置信度调整';
    if(model.id==='yoloe26_reference')return '新版 26 类 · 参考图辅助';
    if(model.id==='yoloe24')return '原始 24 类 · 历史配置';
    return model.class_count===3?'纸、塑料、金属三类专项':'公开 '+model.class_count+' 类';
  }
  function initialTab(saved,available) {
    return saved&&available.includes(saved)?saved:'tab-live';
  }
  if(typeof module!=='undefined')module.exports={groupModels,initialTab};
  if(typeof window==='undefined')return;

  window.addEventListener('load',()=>{
    const create=(tag,text,cls)=>{const n=document.createElement(tag);if(text)n.textContent=text;if(cls)n.className=cls;return n;};
    const style=create('style');
    style.textContent=[
      '#model-switcher{position:relative;z-index:20;display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:12px 0}',
      '#model-choose{display:inline-flex;align-items:center;gap:12px;padding:10px 14px;background:#1c2d39;border:1px solid #476659;border-radius:10px;font-weight:600;text-align:left;max-width:100%}',
      '#model-current{overflow-wrap:anywhere}',
      '#model-choose[aria-expanded="true"]{background:#294138;border-color:#9bdbba}',
      '#model-switch-status{flex-basis:100%;margin:0;color:#b4c9bf;font-size:13px;line-height:1.6}',
      '#model-menu{position:absolute;top:46px;left:0;box-sizing:border-box;width:min(640px,calc(100vw - 52px));padding:8px;border:1px solid #486657;border-radius:14px;background:#17232e;box-shadow:0 16px 48px #0009;color:#edf4ef}',
      '#model-menu[hidden]{display:none!important}',
      '.model-menu-columns{display:grid;grid-template-columns:190px minmax(0,1fr);gap:8px}',
      '#model-families{padding-right:8px;border-right:1px solid #334b40}',
      '#model-config-title{padding:8px 10px;color:#b3c9bd;font-size:12px}',
      '#model-menu button{display:flex;align-items:center;justify-content:space-between;gap:10px;width:100%;padding:11px 10px;border:0;border-radius:8px;background:transparent;color:inherit;text-align:left;font:inherit;white-space:normal}',
      '#model-menu button:hover,#model-menu button:focus-visible,#model-menu .model-family[aria-expanded="true"]{background:#2b4038;outline-color:#a5dfbd}',
      '#model-menu button:disabled{opacity:.4;cursor:default}',
      '#model-menu button strong{display:block;font-size:14px;font-weight:600}',
      '#model-menu button span{display:block;color:#abc4b7;font-size:12px;line-height:1.5;margin-top:3px}',
      '#model-menu .model-check{color:#a5dfbd;font-size:17px;flex-shrink:0}',
      '#model-menu-message{margin:8px 2px 0;padding:10px 8px 4px;border-top:1px solid #334b40;font-size:12px;color:#b4c9bf;line-height:1.5}',
      '@media(max-width:540px){#model-menu{width:calc(100vw - 28px)}.model-menu-columns{grid-template-columns:minmax(115px,.8fr) minmax(0,1.4fr)}#model-menu button{padding:10px 7px}#model-menu button strong{font-size:13px}}',
    ].join('\n');
    document.head.append(style);
    const strip=create('section');strip.id='model-switcher';strip.setAttribute('aria-label','识别模型');
    const open=create('button');open.id='model-choose';open.type='button';open.disabled=true;
    open.setAttribute('aria-label','切换识别模型');open.setAttribute('aria-haspopup','menu');open.setAttribute('aria-expanded','false');open.setAttribute('aria-controls','model-menu');
    const current=create('span','正在读取模型…');current.id='model-current';
    const chevron=create('span','⌄');chevron.setAttribute('aria-hidden','true');open.append(current,chevron);
    const note=create('p');note.id='model-switch-status';note.setAttribute('aria-live','polite');
    strip.append(open,note);
    const nav=document.querySelector('.view-tabs');(nav||document.body.firstChild).before(strip);

    const menu=create('div');menu.id='model-menu';menu.hidden=true;menu.setAttribute('aria-label','模型切换菜单');
    const columns=create('div',null,'model-menu-columns');
    const familyList=create('div');familyList.id='model-families';familyList.setAttribute('role','menu');familyList.setAttribute('aria-label','模型型号');
    const pane=create('div'),title=create('div');title.id='model-config-title';
    const configurations=create('div');configurations.id='model-configurations';configurations.setAttribute('role','menu');configurations.setAttribute('aria-label','模型配置');
    const message=create('p');message.id='model-menu-message';message.setAttribute('role','status');
    pane.append(title,configurations);columns.append(familyList,pane);menu.append(columns,message);strip.append(menu);
    let status,submitting=false,statusAvailable=false,catalogKey='',seenRevision,activeFamily=null,requestError='';

    function closeMenu(focus=false) {
      menu.hidden=true;open.setAttribute('aria-expanded','false');if(focus)open.focus();
    }
    function showConfigurations(id) {
      const group=groupModels(status.models).find(item=>item.id===id);if(!group)return;
      if(activeFamily===id&&configurations.children.length)return;
      activeFamily=id;title.textContent=group.name+' · 配置';
      configurations.replaceChildren();
      for(const button of familyList.children)button.setAttribute('aria-expanded',String(button.dataset.family===id));
      for(const model of group.models) {
        const button=create('button',null,'model-variant');button.id='model-config-'+model.id;button.type='button';
        button.setAttribute('role','menuitemradio');button.setAttribute('aria-checked',String(model.id===status.selected));
        button.dataset.available=String(model.available);
        const text=create('div');text.append(create('strong',model.name),create('span',model.available?modelDetail(model):model.disabled_reason||'暂不可用'));
        button.append(text);if(model.id===status.selected)button.append(create('b','✓','model-check'));
        button.addEventListener('click',()=>selectModel(model));configurations.append(button);
      }
      updateControls();
    }
    function renderMenu() {
      if(!status)return;
      const groups=groupModels(status.models);
      const selectedGroup=groups.find(group=>group.models.some(model=>model.id===status.selected));
      const target=groups.some(group=>group.id===activeFamily)?activeFamily:selectedGroup?.id;
      familyList.replaceChildren();configurations.replaceChildren();activeFamily=null;
      for(const family of groups) {
        const button=create('button',null,'model-family');button.type='button';button.id='model-family-'+family.id;
        button.dataset.family=family.id;button.setAttribute('role','menuitem');button.setAttribute('aria-haspopup','menu');
        const text=create('div');text.append(create('strong',family.name),create('span',family.description));
        button.append(text,create('b','›'));button.addEventListener('pointerenter',()=>showConfigurations(family.id));
        button.addEventListener('click',()=>showConfigurations(family.id));familyList.append(button);
      }
      showConfigurations(target);updateControls();
    }
    function updateControls() {
      if(!status)return;
      const busy=submitting||status.switching;open.disabled=busy||!statusAvailable;
      for(const button of configurations.children)button.disabled=busy||!statusAvailable||button.dataset.available!=='true';
      message.textContent=busy?'正在切换，核对完成后自动刷新。':(status.error||requestError)?'切换失败：'+(status.error||requestError):
        '鼠标移到型号查看配置，点击配置即可切换。✓ 为当前配置。';
    }
    function render(data) {
      const previous=status?.selected;status=data;statusAvailable=true;
      if(seenRevision!==undefined&&seenRevision!==data.revision) {
        const active=document.querySelector('[role="tab"][aria-selected="true"]');
        if(active)sessionStorage.setItem('model-switch-active-tab',active.id);
        location.reload();return;
      }
      seenRevision=data.revision;current.textContent=data.selected_name;
      note.textContent=data.switching?(data.model_motion_follow?'正在切换识别与抓取模型，完成后自动刷新。':'正在加载并核对现场识别结果，完成后自动刷新。'):
        (data.error||requestError)?'切换失败：'+(data.error||requestError)+' 当前仍为 '+data.selected_name+'。':
        data.motion_model_matches?(data.model_motion_follow?'识别与抓取已同步使用当前模型。':'运动按 PYNQ 现有绑定和现场核验执行。'):
        data.model_motion_follow?'模型与抓取配置尚未同步，暂不能开始。':
        '当前用于识别对比；预演和抓取暂停。执行绑定：'+data.bound_name+'。';
      const config=data.selected_configuration;
      if(!data.switching&&config?.confidence!==null&&config?.confidence!==undefined)
        note.textContent+=' 置信度 '+config.confidence.toFixed(2)+'；'+(config.reference_count?'参考图 '+config.reference_count+' 张，原始标签与辅助材质分别保留。':'无参考图辅助。');
      const key=JSON.stringify(data.models);
      if(key!==catalogKey||previous!==data.selected){catalogKey=key;renderMenu();}
      updateControls();
      const tab=document.getElementById('tab-prompted');if(tab)tab.textContent=data.motion_model_matches?'模型参数与背景':'模型识别详情';
      const panel=document.getElementById('view-prompted');
      if(panel&&!data.motion_model_matches) {
        const caption=panel.querySelector('p');if(caption)caption.textContent='点击顶部模型名称，在菜单中选择型号与配置。识别对比保留原标定、投放区和历史任务。';
        const frame=panel.querySelector('iframe');if(frame)frame.title='当前模型识别对比';
      }
      document.title='可回收物中控台 · '+data.selected_name;
      if(!data.motion_model_matches||data.switching)
        for(const id of ['operation-preview','operation-start','operation-arm','operation-voice-arm']) {
          const button=document.getElementById(id);if(button)button.disabled=true;
        }
    }
    async function selectModel(model) {
      if(!status||submitting||status.switching||!statusAvailable||!model.available)return;
      if(model.id===status.selected){closeMenu(true);return;}
      submitting=true;requestError='';closeMenu();render(status);
      try {
        const response=await fetch('/model-switch/select',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({model_id:model.id,revision:status.revision,token:status.token})});
        const data=await response.json();if(!response.ok)throw Error(data.error||'切换未生效');
        submitting=false;render(data);
      } catch(error){submitting=false;requestError=error.message;render(status);}
    }
    open.addEventListener('click',()=>{
      if(!menu.hidden){closeMenu();return;}
      renderMenu();menu.hidden=false;open.setAttribute('aria-expanded','true');
    });
    document.addEventListener('pointerdown',event=>{if(!strip.contains(event.target))closeMenu();});
    document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!menu.hidden){closeMenu(true);event.preventDefault();}});
    function keyboardMenu(container,side) {
      container.addEventListener('keydown',event=>{
        const items=Array.from(container.children).filter(button=>!button.disabled);
        const at=items.indexOf(document.activeElement);
        if(event.key==='ArrowDown'||event.key==='ArrowUp') {
          event.preventDefault();const next=(at+(event.key==='ArrowDown'?1:-1)+items.length)%items.length;items[next]?.focus();
        } else if(event.key==='ArrowRight'&&side==='family') {
          event.preventDefault();const family=document.activeElement?.dataset.family;if(family)showConfigurations(family);
          Array.from(configurations.children).find(button=>!button.disabled)?.focus();
        } else if(event.key==='ArrowLeft'&&side==='config') {
          event.preventDefault();Array.from(familyList.children).find(button=>button.dataset.family===activeFamily)?.focus();
        }
      });
    }
    keyboardMenu(familyList,'family');keyboardMenu(configurations,'config');
    async function poll() {
      try {
        const response=await fetch('/model-switch/status',{cache:'no-store'});
        if(!response.ok)throw Error('模型列表暂不可用');render(await response.json());
      } catch(error){statusAvailable=false;updateControls();note.textContent=error.message;message.textContent=error.message;}
    }
    // Keep the original live-grid: camera/recognition and operation controls
    // remain together. Only change the default tab, never submit an operation.
    const live=document.getElementById('tab-live');
    if(live){live.textContent='控制台';if(nav)nav.prepend(live);}
    const prompted=document.getElementById('view-prompted');
    if(prompted){const jump=prompted.querySelector('button');if(jump)jump.textContent='返回控制台';}
    const saved=sessionStorage.getItem('model-switch-active-tab');sessionStorage.removeItem('model-switch-active-tab');
    const available=Array.from(document.querySelectorAll('[role="tab"]'),tab=>tab.id);
    document.getElementById(initialTab(saved,available))?.click();
    poll();setInterval(poll,1000);
  });
})();

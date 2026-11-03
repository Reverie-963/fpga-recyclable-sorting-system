/* Reusable physical measurements; image coordinates always come from this frame. */
(function(root){
  'use strict';
  const clone=x=>JSON.parse(JSON.stringify(x));
  function features(pixels,runs,size){
    const [w,h]=size,n=w*h,mask=new Uint8Array(n);let offset=0,count=0,sx=0,sy=0,sxx=0,syy=0,sxy=0;
    for(let r=0;r<runs.length;r++){const length=runs[r];if(!Number.isInteger(length)||length<0||offset+length>n)throw Error('参考轮廓无效');
      if(r%2)for(let i=offset;i<offset+length;i++){mask[i]=1;const x=i%w,y=Math.floor(i/w);count++;sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;}offset+=length;}
    if(offset!==n||count<4||pixels.length!==n*4)throw Error('参考图与轮廓不匹配');
    const red=[],green=[],blue=[],hist=Array(12).fill(0),step=Math.max(1,Math.floor(count/2048));let seen=0,samples=0;
    for(let i=0;i<n;i++)if(mask[i]&&seen++%step===0){const x=i%w,y=Math.floor(i/w);
      if(count>30&&(x===0||y===0||x===w-1||y===h-1||!mask[i-1]||!mask[i+1]||!mask[i-w]||!mask[i+w]))continue;
      const r=pixels[i*4]/255,g=pixels[i*4+1]/255,b=pixels[i*4+2]/255;
      red.push(r);green.push(g);blue.push(b);const max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min;
      let hue=d===0?0:max===r?((g-b)/d)%6:max===g?(b-r)/d+2:(r-g)/d+4;hue=(hue+6)%6;
      hist[Math.min(11,Math.floor(hue*2))]++;samples++;
    }
    if(!samples)throw Error('参考轮廓过细，请重新识别');
    const median=values=>values.sort((a,b)=>a-b)[Math.floor(values.length/2)];
    const xx=sxx/count-(sx/count)**2,yy=syy/count-(sy/count)**2,xy=sxy/count-sx*sy/count**2;
    const delta=Math.sqrt((xx-yy)**2+4*xy*xy),large=(xx+yy+delta)/2,small=(xx+yy-delta)/2;
    return {rgb:[median(red),median(green),median(blue)],hist:hist.map(v=>v/samples),aspect:Math.min(1000,Math.sqrt(Math.max(large,.1)/Math.max(small,.1))),area:count/n};
  }
  function score(profile,query,material,model){
    const f=profile.features;if(!f||!query||profile.material!==material||profile.reference_model!==model)return null;
    const area=query.area/f.area,aspect=Math.max(query.aspect/f.aspect,f.aspect/query.aspect);
    const color=Math.sqrt(f.rgb.reduce((v,x,i)=>v+(x-query.rgb[i])**2,0)/3);
    const hist=f.hist.reduce((v,x,i)=>v+Math.abs(x-query.hist[i]),0)/2;
    if(area<.85||area>1.18||aspect>1.2||color>.10||hist>.2)return null;
    return color*.6+hist*.3+Math.abs(Math.log(area))*.1;
  }
  function match(profiles,query,material,model){
    const rows=profiles.map(profile=>({profile,score:score(profile,query,material,model)})).filter(r=>r.score!==null).sort((a,b)=>a.score-b.score);
    if(!rows.length||(rows.length>1&&rows[1].score-rows[0].score<.04))return null;
    return rows[0].profile;
  }
  class Client{
    constructor(request){this.request=request;this.data={revision:0,profiles:[]};}
    async load(){const data=await this.request('/api/item-profiles');if(data.schema!=='item-profiles-v1'||!Array.isArray(data.profiles))throw Error('物品档案暂不可用');this.data=data;return data;}
    async save(profile,id){this.data=await this.request('/api/item-profiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'save',expected_revision:this.data.revision,...(id?{id}:{}),profile})});return this.data.profiles.find(p=>p.id===id)||this.data.profiles[this.data.profiles.length-1];}
    async remove(id){this.data=await this.request('/api/item-profiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'delete',expected_revision:this.data.revision,id})});}
  }
  root.SortingItemProfiles={features,match,score,Client,clone};
  if(typeof module!=='undefined')module.exports=root.SortingItemProfiles;
})(typeof window!=='undefined'?window:globalThis);

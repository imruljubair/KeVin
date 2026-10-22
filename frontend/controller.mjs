import {uiStore, actions} from './ui-store.mjs';
import {colorRamp, cameraTransform, cameraEase, applySceneTransform, svgNode} from './visualization.mjs';
export function startController(){
'use strict';
const $ = id => document.getElementById(id);
const DEMO = window.KevinDemo || null;
const state = {status:null, trace:null, busy:false, playing:false, validated:null, zoom:.6, x:10, y:5, width:0, height:0, blocks:new Map(), selected:null, showProjections:true};
const colors = {k:[138,219,139],kt:[138,219,139],kc:[138,219,139],v:[181,94,184],vc:[181,94,184],q:[135,145,155],a:[0,181,226],scores:[0,181,226],va:[241,215,83],x:[241,215,83],wq:[242,163,104],wk:[242,163,104],wv:[242,163,104]};
const descriptions = {
  wq:['Wq · query weights','The selected query head’s real projection weights. Values are dequantized from GGUF storage; omitted components remain hidden.'],
  wk:['Wk · key weights','The selected KV head’s real projection weights. Query heads in the same GQA group share these weights.'],
  wv:['Wv · value weights','The selected KV head’s real value projection weights, dequantized from GGUF storage.'],
  x:['Attention input','The normalized hidden states entering this layer’s attention. The complete model runs underneath this view.'],
  k:['Keys · after RoPE','The new token’s key projection after rotary position encoding, read after its F16 cache write. Each group of query heads shares one KV head.'],
  v:['Values','The new value projection, read after its F16 cache write. Values hold the information that attention will mix into its output.'],
  q:['Queries · after RoPE','The current token’s query projection after rotary position encoding. Queries are compared with all visible cached keys.'],
  kt:['Kᵀ · cached keys','The actual F16 key-cache operands, transposed for attention. Each row corresponds to an earlier token, including the current token.'],
  a:['A · attention probabilities','Actual softmax probabilities for this query head. Future positions are masked during prefill; decode attends only to the prefix and current token.'],
  va:['VA · attention output','The model’s actual weighted mixture of cached values for this head, before the attention output projection.'],
  kc:['K cached','The actual key cache after this step. Earlier entries are reused. New entries are appended, and KeVin verifies that the existing prefix stays unchanged.'],
  vc:['V cached','The actual value cache after this step. Colors reflect values read back from the F16 runtime cache, including its storage precision.']
};
descriptions.scores=['Scores · raw attention','Scaled dot products of the captured Kᵀ rows and Q columns, before causal masking and column-wise softmax.'];
const labels={scores:'Scores',wq:'Wq · query weights',wk:'Wk · key weights',wv:'Wv · value weights',x:'X · normalized input',k:'K · after RoPE',v:'V · projection',q:'Q · after RoPE',kt:'Kᵀ',a:'A · softmax',va:'VA',kc:'K cached',vc:'V cached'};
const tokenTooltip=text('div','timeline-tooltip');tokenTooltip.hidden=true;tokenTooltip.setAttribute('role','tooltip');document.body.append(tokenTooltip);
let tokenHoverTimer=0;
function hideTokenTooltip(){clearTimeout(tokenHoverTimer);tokenTooltip.hidden=true;}
function showTokenTooltip(header){
  hideTokenTooltip();tokenHoverTimer=setTimeout(()=>{
    tokenTooltip.replaceChildren(text('div','',header.dataset.hoverTitle));
    const tokens=text('div','timeline-token-boxes');for(const token of JSON.parse(header.dataset.tokens||'[]'))tokens.append(text('span','token-chip',tokenText(token)));tokenTooltip.append(tokens);
    tokenTooltip.classList.toggle('decode-token-tooltip',header.dataset.decode==='true');tokenTooltip.hidden=false;
    const r=header.getBoundingClientRect(),box=tokenTooltip.getBoundingClientRect();tokenTooltip.style.left=Math.max(8,Math.min(innerWidth-box.width-8,(r.left+r.right-box.width)/2))+'px';tokenTooltip.style.top=(r.bottom+5)+'px';
  },450);
}
let toastTimer;
function toast(message){uiStore.set({toast:message});clearTimeout(toastTimer);toastTimer=setTimeout(()=>uiStore.set({toast:''}),6500);}
async function api(path,body){if(DEMO)return DEMO.request(path,body);const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data;}
function text(tag,cls,content){const n=document.createElement(tag);if(cls)n.className=cls;if(content!==undefined)n.textContent=content;return n;}
function position(node,x,y,w,h){Object.assign(node.style,{left:x+'px',top:y+'px'});if(w!==undefined)node.style.width=w+'px';if(h!==undefined)node.style.height=h+'px';}
function bytes(n){return n>=1024**3?(n/1024**3).toFixed(1)+' GiB':n>=1024**2?(n/1024**2).toFixed(0)+' MiB':(n/1024).toFixed(1)+' KiB';}
function setBusy(value){state.busy=value;refreshControls();}
function refreshControls(){
  const ready=state.status?.phase==='ready',t=state.trace;
  uiStore.set({ready,busy:state.busy,playing:state.playing,
    canStep:Boolean(ready&&t?.steps.length&&!t.ended&&!t.full),hasTrace:Boolean(t?.steps.length),
    demoLimit:Boolean(DEMO?.atLimit())});
}
function applyStatus(status){
  state.status=status;
  uiStore.set({status:status.message,modelName:status.model?.name||'Choose a model',
    layerCount:status.model?.layers||0,headCount:status.model?.heads||0});
  updateEmptyState();refreshControls();
}
async function getStatus(){const s=await api('/api/status');applyStatus(s);return s;}
function updateEmptyState(){
  document.body.classList.toggle('has-trace',Boolean(state.trace?.steps.length));
  const phase=state.status?.phase,ready=phase==='ready',loading=['loading','validating','downloading'].includes(phase);
  uiStore.set({emptyTitle:loading?'Loading model…':ready?'Enter a prompt':'Choose a model',
    emptyDescription:loading?state.status.message:ready?'Click Prefill above to view the matrices.':'Load a model to inspect its KV cache.',
    emptyActionHidden:ready||loading,hasTrace:Boolean(state.trace?.steps.length)});
  if(!state.trace?.steps.length)for(const id of ['memory-panel','memory-toggle','memory-resize','memory-links','decode-group'])$(id).hidden=true;
}
function resetDiagram(){
  closeInspector();state.playing=false;state.trace=null;state.blocks.clear();memory.count=0;
  for(const id of ['scene','axis-labels','step-labels','memory-panel','memory-toggle','memory-resize','memory-links','decode-group'])$(id).hidden=true;
  uiStore.set({hasTrace:false,metrics:''});updateEmptyState();
}
function openModel(){if(DEMO){DEMO.localOnly('Run KeVin locally to load a different model.');return;}$('model-dialog').showModal();}
actions.openModel=openModel;
function resetValidation(){state.validated=null;uiStore.set({validation:'',loadLabel:'Load model'});}
actions.resetValidation=resetValidation;
function validationError(message){uiStore.set({validation:message});}
actions.loadModel=async()=>{
  const choice=$('model-choice'),custom=choice.value==='custom';
  const url=custom?$('model-url').value.trim():choice.value==='recommended'?state.status?.default_url:choice.selectedOptions[0].dataset.url;
  if(!url){validationError('Enter a GGUF file link.');$('model-url').focus();return;}
  uiStore.set({modelLoading:true,validation:''});setBusy(true);
  let loaded=false;
  try{
    uiStore.set({loadLabel:'Checking model…'});
    const info=await api('/api/inspect',{url});
    resetDiagram();uiStore.set({loadLabel:'Loading model…',progressVisible:true,progress:0,progressMessage:'Preparing download…'});
    await api('/api/load',{url:info.url});
    let status;
    do{
      await new Promise(resolve=>setTimeout(resolve,600));status=await getStatus();
      uiStore.set({progress:status.progress,progressMessage:status.message});
    }while(!['ready','error'].includes(status.phase));
    if(status.phase==='error')throw Error(status.message);
    $('model-dialog').close();resetValidation();loaded=true;
  }catch(error){validationError(error.message);uiStore.set({loadLabel:'Load model'});}
  finally{uiStore.set({modelLoading:false,progressVisible:false});setBusy(false);updateEmptyState();if(loaded)$('prompt').focus();}
};
function tokenText(s){return s.replaceAll(' ','␣').replaceAll('\n','↵').replaceAll('\t','⇥')||'∅';}
async function inference(command){setBusy(true);uiStore.set({status:command.op==='prefill'?'Processing prompt · capturing real activations…':command.op==='step'?'Decoding one token · appending K/V…':'Switching the inspected layer/head…'});
  try{const trace=await api('/api/session',command);state.trace=trace;state.animateAppend=command.op==='step';render();state.animateAppend=false;return trace;}
  catch(error){resetDiagram();toast(error.message);uiStore.set({status:error.message});throw error;}
  finally{setBusy(false);}
}
async function prefill(){if(state.busy||state.playing)return;closeInspector();try{await inference({op:'prefill',prompt:$('prompt').value});overview({animate:false});}catch{}}
actions.prefill=prefill;
actions.step=async()=>{if(DEMO?.atLimit()){DEMO.localOnly('You’ve reached the recorded token limit. Run KeVin locally to generate more tokens.');return;}try{await inference({op:'step'});}catch{}};
actions.play=async()=>{
  if(DEMO?.atLimit()){DEMO.localOnly('Run KeVin locally to generate beyond the recorded token limit.');return;}
  if(state.playing){state.playing=false;refreshControls();return;}
  state.playing=true;refreshControls();
  try{while(state.playing){const t=await inference({op:'step'});if(t.ended||t.full||DEMO?.atLimit())break;await new Promise(r=>setTimeout(r,220));}}
  catch{}finally{state.playing=false;refreshControls();}
};
actions.view=async()=>{const {layer,head}=uiStore.getSnapshot();try{await inference({op:'view',layer,head});}catch{}};
actions.compact=()=>{if(state.trace){render();overview();}};
actions.numbers=()=>{applyCamera();};
const svgElement=svgNode;
// Calibrate V once from the prefill cache, keeping equal values the same color
// across projections, cached operands, memory, and later generated tokens.
let valueColorScale=1,scoreColorScale=1;
const activationColorScales={};
function calibrateValueColors(trace){
  const magnitudes=trace.steps[0].matrices.vc.flat().filter(Number.isFinite).map(Math.abs).sort((a,b)=>a-b);
  valueColorScale=magnitudes[Math.floor((magnitudes.length-1)*.95)]||1;
  const scores=KavinProxy.scoreValues(trace.steps[0]);
  const scoreMagnitudes=scores.flatMap((row,r)=>row.filter((value,c)=>r<=c&&Number.isFinite(value)).map(Math.abs)).sort((a,b)=>a-b);
  scoreColorScale=scoreMagnitudes[Math.floor((scoreMagnitudes.length-1)*.95)]||1;
  for(const key of ['q','x','va']){
    const values=trace.steps[0].matrices[key].flat().filter(Number.isFinite).map(Math.abs).sort((a,b)=>a-b);
    activationColorScales[key]=values[Math.floor((values.length-1)*.95)]||1;
  }
}
function shade(value,key){
  if(key==='v'||key==='vc'){
    const base=colors.v,neutral=[244,230,247];
    const intensity=Math.pow(Math.min(1,Math.abs(value)/valueColorScale),.65);
    return colorRamp(neutral,base,intensity);
  }
  const base=colors[key],neutral=base.map(v=>Math.round(255*.88+v*.12));
  const intensity=key in activationColorScales?Math.pow(Math.min(1,Math.abs(value)/activationColorScales[key]),.8):key==='scores'?Math.pow(Math.min(1,Math.abs(value)/scoreColorScale),.8):key==='a'?Math.sqrt(Math.max(0,Math.min(1,value))):Math.abs(Math.tanh(value/.35));
  return colorRamp(neutral,base,intensity);
}
function masked(block,item){return block.causal&&item.rowRange[0]>item.colRange[1];}
function cellInfo(block,event){
  const rect=block.node.getBoundingClientRect();
  return KavinProxy.hitTest(block.plan,(event.clientX-rect.left)*block.w/rect.width,(event.clientY-rect.top)*block.h/rect.height);
}
function rangeLabel(range){return range[0]===range[1]?String(range[0]+1):`${range[0]+1}–${range[1]+1}`;}
function omittedInfo(item){return `Rows ${rangeLabel(item.rowRange)}, columns ${rangeLabel(item.colRange)} are omitted. This shaded proxy region has no displayed numerical value.`;}
function inspectCell(block,item){
  if(!item)return;
  $('clear-focus').hidden=false;
  block.node.classList.add('has-cell-focus');
  state.selectedRegion=item;drawConnections(block.index,block.key,item);
  const out=$('inspect-value');out.hidden=false;
  if(item.omitted){state.selectedValue=null;out.textContent=omittedInfo(item);return;}
  state.selectedValue={key:state.selected,row:item.row,col:item.col};
  out.textContent=`Row ${item.row+1}, column ${item.col+1}: ${String(block.values[item.row][item.col])}${masked(block,item)?' · causally masked':''}`;
}
const omissionDotTiles=new Map();
function omissionDotPattern(ctx,scale=1,x=0,y=0){
  if(!omissionDotTiles.has(scale)){
    const tile=document.createElement('canvas');tile.width=tile.height=12*scale;
    const pen=tile.getContext('2d');pen.fillStyle='rgba(136,153,173,.32)';
    pen.beginPath();pen.arc(6*scale,6*scale,.7*scale,0,2*Math.PI);pen.fill();omissionDotTiles.set(scale,tile);
  }
  const pattern=ctx.createPattern(omissionDotTiles.get(scale),'repeat');
  pattern.setTransform(new DOMMatrix().translate(x,y));return pattern;
}
// Cache the feathered star mask by footprint; decoding reuses it unchanged.
const omissionVeils=new Map();
function omissionVeil(shape,p){
  const width=Math.round(shape.rx*p.cellWidth*4),height=Math.round(shape.ry*p.cellHeight*4),key=`${width}:${height}`;
  if(omissionVeils.has(key))return omissionVeils.get(key);
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(width,height);
  const xs=Array.from({length:width},(_,x)=>Math.pow(Math.abs((x+.5-width/2)/(width/2)),KavinProxy.OMISSION_POWER));
  const ys=Array.from({length:height},(_,y)=>Math.pow(Math.abs((y+.5-height/2)/(height/2)),KavinProxy.OMISSION_POWER));
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=(y*width+x)*4;pixels.data[i]=237;pixels.data[i+1]=241;pixels.data[i+2]=246;
    pixels.data[i+3]=Math.round(255*(1-KavinProxy.omissionFade(xs[x]+ys[y])));
  }
  ctx.putImageData(pixels,0,0);
  ctx.globalCompositeOperation='source-atop';ctx.fillStyle=omissionDotPattern(ctx,2);ctx.fillRect(0,0,width,height);
  ctx.globalCompositeOperation='source-over';
  if(omissionVeils.size>=12)omissionVeils.delete(omissionVeils.keys().next().value);
  omissionVeils.set(key,canvas);return canvas;
}
function featherGapEdges(ctx,block){
  const p=block.plan;ctx.save();ctx.beginPath();
  for(let r=0;r<p.rowIndices.length;r++)for(let c=0;c<p.colIndices.length;c++){
    const item=KavinProxy.cellAt(p,r,c);
    if(!masked(block,item))ctx.rect(p.padX+c*p.cellWidth,p.padY+r*p.cellHeight,p.cellWidth,p.cellHeight);
  }
  ctx.clip();
  for(const vertical of [false,true]){
    const indices=vertical?p.rowIndices:p.colIndices,gap=indices.indexOf(null);if(gap<0)continue;
    const size=vertical?p.cellHeight:p.cellWidth,pad=vertical?p.padY:p.padX,feather=size*.85;
    for(const side of [-1,1]){
      const edge=pad+(side<0?gap:gap+1)*size,outer=edge+side*feather;
      const gradient=ctx.createLinearGradient(vertical?0:outer,vertical?outer:0,vertical?0:edge,vertical?edge:0);
      for(let i=0;i<=12;i++){const t=i/12;gradient.addColorStop(t,`rgba(237,241,246,${t*t*(3-2*t)})`);}
      ctx.fillStyle=gradient;
      if(vertical)ctx.fillRect(p.padX,Math.min(edge,outer),p.colIndices.length*p.cellWidth,feather);
      else ctx.fillRect(Math.min(edge,outer),p.padY,feather,p.rowIndices.length*p.cellHeight);
    }
  }
  ctx.restore();
}
let matrixPointer=null;
function isSelectedCell(block,item){
  const previous=state.selectedRegion;
  return Boolean(item&&state.selected===`${block.index}:${block.key}`&&previous&&item.rowRange.every((v,i)=>v===previous.rowRange[i])&&item.colRange.every((v,i)=>v===previous.colRange[i]));
}
function updateMatrixCursor(){
  if(!matrixPointer)return;
  const {node,event}=matrixPointer,b=node.blockData;
  const item=cellInfo(b,event);
  node.classList.toggle('over-selected-cell',isSelectedCell(b,item));
  node.classList.toggle('over-matrix-padding',!item);
}
function drawBlock(block,column){
  let node=column.querySelector(`[data-key="${block.key}"]`);
  if(!node){
    node=text('button','matrix');node.type='button';node.dataset.key=block.key;
    node.append(document.createElement('canvas'),text('span','matrix-name'),text('span','matrix-shape'),svgElement('svg',{class:'matrix-annotations','aria-hidden':'true'}));
    const hover=text('span','cell-hover');hover.hidden=true;hover.setAttribute('aria-hidden','true');node.append(hover);
    column.append(node);
    node.onclick=event=>{
      event.stopPropagation();if(event.detail&&panMoved||spacePan)return;
      const b=node.blockData,item=event.detail?cellInfo(b,event):null;
      if(event.detail&&!item)return;
      if(isSelectedCell(b,item)){stopCamera();closeInspector();updateMatrixCursor();return;}
      if(state.zoom<.7-1e-6){focusMatrix(b,.7);return;}
      stopCamera();selectMatrix(b);inspectCell(b,item);
      updateMatrixCursor();
    };
    node.onpointermove=event=>{
      const b=node.blockData,item=cellInfo(b,event),p=b.plan;
      matrixPointer={node,event:{clientX:event.clientX,clientY:event.clientY}};updateMatrixCursor();
      hover.hidden=Boolean(drag||!item||item.omitted||masked(b,item));
      if(!hover.hidden)position(hover,p.padX+p.colIndices.indexOf(item.col)*p.cellWidth,p.padY+p.rowIndices.indexOf(item.row)*p.cellHeight,p.cellWidth,p.cellHeight);
      node.title=drag?'':state.zoom<.7-1e-6?'Click to zoom into this matrix':!item?node.getAttribute('aria-label'):item.omitted?omittedInfo(item):`Row ${item.row+1}, column ${item.col+1}: ${String(b.values[item.row][item.col])}${masked(b,item)?' · causally masked':''}`;
    };
    node.onpointerleave=()=>{hover.hidden=true;node.classList.remove('over-selected-cell','over-matrix-padding');if(matrixPointer?.node===node)matrixPointer=null;node.title=node.getAttribute('aria-label');};
  }
  block.node=node;node.blockData=block;
  const p=block.plan,canvas=node.querySelector('canvas');
  canvas.width=Math.ceil(block.w*2);canvas.height=Math.ceil(block.h*2);canvas.style.width=block.w+'px';canvas.style.height=block.h+'px';
  const ctx=canvas.getContext('2d');ctx.scale(2,2);
  const diamond=KavinProxy.cellAt(p,1,1)?.diamond;
  for(let ri=0;ri<p.rowIndices.length;ri++)for(let ci=0;ci<p.colIndices.length;ci++){
    const item=KavinProxy.cellAt(p,ri,ci);if(masked(block,item))continue;
    const x=p.padX+ci*p.cellWidth,y=p.padY+ri*p.cellHeight;
    ctx.fillStyle='#edf1f6';ctx.fillRect(x,y,p.cellWidth,p.cellHeight);
    if(item.row===null||item.col===null){ctx.fillStyle=omissionDotPattern(ctx,1,p.padX,p.padY);ctx.fillRect(x,y,p.cellWidth,p.cellHeight);}
    else{
      ctx.globalAlpha=item.diamond?1:item.fade;ctx.fillStyle=shade(block.values[item.row][item.col],block.kind);ctx.fillRect(x,y,p.cellWidth,p.cellHeight);ctx.globalAlpha=1;
    }
    if(!item.omitted){ctx.strokeStyle=`rgba(255,255,255,${.66*item.fade})`;ctx.lineWidth=.7;ctx.strokeRect(x+.35,y+.35,p.cellWidth-.7,p.cellHeight-.7);}

  }
  if(diamond){
    ctx.save();ctx.beginPath();
    // Clip to the original prefill footprint and valid causal cells only.
    for(let r=0;r<p.rowIndices.length;r++)for(let c=0;c<p.colIndices.length;c++){
      const item=KavinProxy.cellAt(p,r,c);
      if(item.diamond&&!masked(block,item))ctx.rect(p.padX+c*p.cellWidth,p.padY+r*p.cellHeight,p.cellWidth,p.cellHeight);
    }
    ctx.clip();
    const veil=omissionVeil(diamond,p);
    ctx.drawImage(veil,p.padX,p.padY,diamond.rx*p.cellWidth*2,diamond.ry*p.cellHeight*2);ctx.restore();
  }
  if(p.rowGap||p.colGap)featherGapEdges(ctx,block);
  // Keep axes separate from data so ellipses never masquerade as matrix values.
  const annotations=node.querySelector('.matrix-annotations');annotations.setAttribute('viewBox',`0 0 ${block.w} ${block.h}`);annotations.replaceChildren();node.annotationStamp=null;
  node.querySelector('.matrix-name').textContent={scores:'Scores',wq:'Wq',wk:'Wk',wv:'Wv',x:'x',k:'K',v:'V',q:'Q',kt:'Kᵀ',a:'A',va:'VA',kc:'K cached',vc:'V cached',v_operand:'V'}[block.key];
  node.querySelector('.matrix-name').title=(block.key==='v_operand'?'V':labels[block.key]);
  node.style.setProperty('--matrix-label-x',(p.padX-(p.rows===1?13:29))+'px');
  node.style.setProperty('--matrix-label-y',(p.padY+p.rowIndices.length*p.cellHeight/2)+'px');
  node.querySelector('.matrix-name').dataset.short={scores:'Scores',wq:'Wq',wk:'Wk',wv:'Wv',x:'x',k:'K',v:'V',q:'Q',kt:'Kᵀ',a:'A',va:'VA',kc:'K cached',vc:'V cached',v_operand:'V'}[block.key];
  const foregroundName=text('button','scene-matrix-name',node.querySelector('.matrix-name').textContent);
  foregroundName.dataset.fullName=foregroundName.textContent;foregroundName.type='button';foregroundName.dataset.matrix=block.key;foregroundName.dataset.step=block.index;
  foregroundName.title=`${foregroundName.textContent} · click to show related matrices`;
  const centered=Boolean(p.rowGap||p.colGap);
  foregroundName.classList.toggle('centered-name',centered);
  const ownEdgeName=['scores','a'].includes(block.key)&&!centered;
  foregroundName.classList.toggle('own-edge-name',ownEdgeName);
  const sideInset=p.rows===1?13:29;
  // Scores and A straddle their own left edge, with most of the label inside
  // that matrix instead of extending toward the preceding K/Q matrix.
  const nameX=centered?p.padX+(p.colGap?p.colIndices.indexOf(null)+.5:p.colIndices.length/2)*p.cellWidth:ownEdgeName?p.padX:p.padX-sideInset;
  const nameY=p.padY+(p.rowGap?p.rowIndices.indexOf(null)+.5:p.rowIndices.length/2)*p.cellHeight;
  position(foregroundName,block.x+nameX,block.y+nameY);
  block.nameNode=foregroundName;
  foregroundName.onclick=event=>{
    event.stopPropagation();
    if(event.detail&&panMoved||spacePan)return;
    if(state.nameSelection&&state.selected===`${block.index}:${block.key}`){closeInspector();return;}
    stopCamera();selectMatrix(block,{names:true});
  };
  $('scene-labels').append(foregroundName);
  node.querySelector('.matrix-shape').textContent=`${p.rows} × ${p.cols}${block.appendOnly?' · appended '+(block.appendAxis==='row'?p.rowIndices.length+' row(s)':p.colIndices.length+' column(s)'):''}${p.rowGap||p.colGap?` · ${p.rowIndices.length} × ${p.colIndices.length} proxy`:''}`;
  node.dataset.rows=p.rows;node.dataset.cols=p.cols;node.dataset.proxyRows=p.rowIndices.length;node.dataset.proxyCols=p.colIndices.length;
  node.dataset.sourceStep=block.sourceStep;node.dataset.shared=String(Boolean(block.shared));
  node.setAttribute('aria-label',`${node.querySelector('.matrix-name').textContent}, ${p.rows} by ${p.cols}. ${p.retained} original entries retained${block.appendOnly?", appended entries only":""}${p.rowGap||p.colGap?', shaded gaps omitted':''}. Open explanation.`);
  node.title=node.getAttribute('aria-label');
  position(node,block.x-state.timeline.columns[block.index].x,block.y,block.w,block.h);
  state.blocks.set(`${block.index}:${block.key}`,block);
}
function drawAnnotations(block,mode){
  const node=block.node,stamp=mode==='values'?`${mode}:${Math.round(state.zoom*20)}`:mode;if(node.annotationStamp===stamp)return;
  const svg=node.querySelector('.matrix-annotations');svg.replaceChildren();node.annotationStamp=stamp;
  const p=block.plan;
  // Worksheet outlines stay legible at overview zoom, including the causal staircase.
  const outlines={solid:'',omitted:''};
  function edge(d,omitted){outlines[omitted?'omitted':'solid']+=d;}
  for(let ri=0;ri<p.rowIndices.length;ri++)for(let ci=0;ci<p.colIndices.length;ci++){
    const item=KavinProxy.cellAt(p,ri,ci);if(masked(block,item))continue;
    const x=p.padX+ci*p.cellWidth,y=p.padY+ri*p.cellHeight,w=p.cellWidth,h=p.cellHeight;
    if(ri===0||masked(block,KavinProxy.cellAt(p,ri-1,ci)))edge(`M${x},${y}h${w}`,item.col===null);
    if(ci===0||masked(block,KavinProxy.cellAt(p,ri,ci-1)))edge(`M${x},${y}v${h}`,item.row===null);
    if(ri===p.rowIndices.length-1||masked(block,KavinProxy.cellAt(p,ri+1,ci)))edge(`M${x},${y+h}h${w}`,item.col===null);
    if(ci===p.colIndices.length-1||masked(block,KavinProxy.cellAt(p,ri,ci+1)))edge(`M${x+w},${y}v${h}`,item.row===null);
  }
  for(const [kind,d] of Object.entries(outlines))if(d){
    const attrs={d,fill:'none',stroke:'#68788a','stroke-width':1/state.zoom,'vector-effect':'none',class:'matrix-outline'+(kind==='omitted'?' omitted-outline':'')};
    if(kind==='omitted')attrs['stroke-dasharray']=`${3/state.zoom} ${3/state.zoom}`;
    svg.append(svgElement('path',attrs));
  }
  if(block.shared&&block.appendStart){const indices=block.appendAxis==='row'?p.rowIndices:p.colIndices,i=indices.indexOf(block.appendStart);if(i>0)svg.append(svgElement('path',{class:'append-boundary',d:block.appendAxis==='row'?`M${p.padX},${p.padY+i*p.cellHeight}h${p.colIndices.length*p.cellWidth}`:`M${p.padX+i*p.cellWidth},${p.padY}v${p.rowIndices.length*p.cellHeight}`,stroke:'#68788a','stroke-width':1/state.zoom,'vector-effect':'none'}));}
  if(!mode)return;
  if(p.cols>1||block.key==='x')for(let ci=0;ci<p.colIndices.length;ci++)svg.append(svgElement('text',{x:p.padX+(ci+.5)*p.cellWidth,y:p.padY-8,'text-anchor':'middle',class:'proxy-axis'},p.colIndices[ci]===null?'…':p.colIndices[ci]+1));
  if(p.rows>1)for(let ri=0;ri<p.rowIndices.length;ri++)svg.append(svgElement('text',{x:p.padX-7,y:p.padY+(ri+.5)*p.cellHeight,'text-anchor':'end','dominant-baseline':'central',class:'proxy-axis'},p.rowIndices[ri]===null?'…':p.rowIndices[ri]+1));
  if(block.key==='x')for(let ci=0;ci<p.colIndices.length;ci++){
    const col=p.colIndices[ci],index=col===null?null:state.trace.steps[block.index].start+col;
    const token=index===null?null:state.trace.tokens[index];
    const x=p.padX+(ci+.5)*p.cellWidth,y=p.padY-23;
    const label=svgElement('text',{x,y,transform:`rotate(-45 ${x} ${y})`,'text-anchor':'start',class:'input-token-label','data-token-index':index??'omitted'},token?tokenText(token.text):'…');
    if(token)label.append(svgElement('title',{},token.text));
    svg.append(label);
  }
  if(mode!=='values')return;
  for(let ri=0;ri<p.rowIndices.length;ri++)for(let ci=0;ci<p.colIndices.length;ci++){
    const item=KavinProxy.cellAt(p,ri,ci);if(item.omitted||masked(block,item))continue;
    const value=svgElement('text',{x:p.padX+(ci+.5)*p.cellWidth,y:p.padY+(ri+.5)*p.cellHeight,'text-anchor':'middle','dominant-baseline':'central',class:'proxy-value',opacity:item.fade,'data-row':item.row,'data-col':item.col},KavinProxy.formatNumber(block.values[item.row][item.col]));
    svg.append(value);
  }
  markFocusedNumbers(block);
}
function render(){
  const t=state.trace;if(!t?.steps.length)return;
  document.body.classList.add('has-trace');
  calibrateValueColors(t);
  uiStore.set({hasTrace:true});$('scene').hidden=false;$('axis-labels').hidden=false;$('step-labels').hidden=false;
  uiStore.set({layer:t.layer,head:t.head,kvHead:t.kv_head});
  const timeline=KavinProxy.layout(t,{compact:$('compact').checked,showProjections:state.showProjections});state.timeline=timeline;
  state.blocks.clear();$('scene-labels').replaceChildren();$('axis-labels').replaceChildren();
  for(const group of timeline.groups){
    const label=text('div','row-label');label.dataset.top=group.top;label.dataset.height=group.height;
    label.classList.toggle('value-aggregation',group.title==='Value aggregation');
    label.append(text('span','vertical-title',group.title));
    const children=text('div','row-children');
    group.labels.forEach((name,index)=>{const child=text('span','row-name',name);child.dataset.offset=group.centers[index]-group.top;children.append(child);});
    label.append(children);$('axis-labels').append(label);
  }
  for(const col of timeline.columns){
    let node=$('columns').children[col.index];
    if(!node){node=text('section','column');if(state.animateAppend)node.classList.add('is-appended');node.append(text('div','token-row'),text('div','kv-step-footer'));$('columns').append(node);}
    position(node,col.x,0,col.width,timeline.height);node.dataset.step=col.index;
    const end=col.step.start+col.step.count;
    let header=$('step-labels').children[col.index];
    if(!header){header=text('div','timeline-header');header.onpointerenter=()=>showTokenTooltip(header);header.onpointerleave=hideTokenTooltip;$('step-labels').append(header);}
    header.replaceChildren(text('strong','',col.index===0?'Prefill':`Decode ${col.index}`),text('small','',col.index===0?`${col.step.count} tokens`:''));
    header.dataset.label=col.index===0?'Prefill':`Decode ${col.index}`;header.dataset.short=col.index===0?'Prefill':`D${col.index}`;
    header.removeAttribute('title');header.dataset.hoverTitle=col.index?`Position ${end}`:`Prefill · ${col.step.count} tokens`;header.dataset.decode=String(col.index>0);header.dataset.tokens=JSON.stringify(t.tokens.slice(col.step.start,end).map(token=>token.text));header.setAttribute('role','group');header.setAttribute('aria-label',header.dataset.label);
    const tokenRow=node.querySelector('.token-row');tokenRow.replaceChildren(...t.tokens.slice(col.step.start,end).map(token=>{const chip=text('span','token-chip'+(col.index?' new':''),tokenText(token.text));chip.title=token.text;return chip;}));
    tokenRow.style.left=col.right+'px';tokenRow.style.maxWidth=(col.width-col.right-30)+'px';
    for(const old of node.querySelectorAll('.matrix'))if(!col.blocks[old.dataset.key])old.remove();
    for(const block of Object.values(col.blocks))drawBlock(block,node);
    for(const old of node.querySelectorAll('.projection-operation'))old.remove();
    for(const kind of ['q','k','v']){const b=col.blocks[kind];if(!b||!state.trace.weights)continue;const label=text('span','projection-operation',state.trace.weights.transforms[kind].replace('linear',`W${kind} × X`));position(label,col.right,b.y-46);node.append(label);}
    const foot=node.querySelector('.kv-step-footer');position(foot,col.right,timeline.height-22);foot.hidden=true;
    foot.textContent=col.index===0?'Cache created':`${col.step.ms.toFixed(0)} ms · cache prefix preserved`;
  }
  while($('columns').children.length>timeline.columns.length)$('columns').lastChild.remove();
  while($('step-labels').children.length>timeline.columns.length)$('step-labels').lastChild.remove();
  state.width=timeline.width;state.height=timeline.height;position($('scene'),0,0,state.width,state.height);
  const generated=t.tokens.length-t.steps[0].count;
  uiStore.set({status:t.ended?'End of sequence reached':t.full?'Context limit reached · reset with a new prompt':`${t.tokens.length} tokens · ${generated} generated${t.cache_prefix_verified?' · cache prefix verified':''}`,
    metrics:`${bytes(t.logical_cache_bytes)} LOGICAL KV · F16 · ${t.model.layers} LAYERS · GREEDY`});
  if(state.selected){const block=state.blocks.get(state.selected);if(block)selectMatrix(block,{names:state.nameSelection});else closeInspector();}
  renderMemory();
  applyCamera();
  if(DEMO)DEMO.afterRender();
}
const memory={selection:null,width:200,count:0,collapsed:false,hoverToken:null};
function memoryBytes(n){return n<1024?`${n} B`:`${(n/1024).toFixed(2).replace(/\.?0+$/,'')} KiB`;}
function highlightMemoryColumn(token){
  if(memory.hoverToken===token)return;
  memory.hoverToken=token;
  $('memory-arrays').querySelectorAll('.memory-column-hover').forEach(node=>node.classList.remove('memory-column-hover'));
  if(token!==null)$('memory-arrays').querySelectorAll(`td[data-token="${token}"],th[data-token="${token}"]`).forEach(node=>node.classList.add('memory-column-hover'));
}
function syncMemoryScroll(source){
  for(const array of $('memory-arrays').querySelectorAll('.memory-array'))if(array!==source&&Math.abs(array.scrollLeft-source.scrollLeft)>.5)array.scrollLeft=source.scrollLeft;
  drawMemoryLinks();
}
function memoryWidth(){return memory.collapsed?28:memory.width;}
function positionCameraToolbar(){
  const toolbar=$('camera-toolbar'),viewport=$('viewport').getBoundingClientRect(),dock=$('memory-panel').getBoundingClientRect(),half=toolbar.offsetHeight/2;
  const center=memory.collapsed?(viewport.top+viewport.bottom)/2:(dock.top+dock.bottom)/2;
  toolbar.style.right=(memory.collapsed?38:memory.width+20)+'px';
  toolbar.style.top=Math.max(viewport.top+half+12,Math.min(viewport.bottom-half-12,center))+'px';
}
function updateMemoryDock(){
  $('memory-panel').hidden=memory.collapsed;$('memory-resize').hidden=memory.collapsed;
  const toggle=$('memory-toggle');toggle.hidden=false;toggle.classList.toggle('is-collapsed',memory.collapsed);
  toggle.textContent=memory.collapsed?'‹ KV memory':'›';
  toggle.title=memory.collapsed?'Show KV tensor memory':'Hide memory';
  toggle.setAttribute('aria-expanded',String(!memory.collapsed));
  toggle.setAttribute('aria-label',memory.collapsed?'Show KV tensor memory':'Collapse KV tensor memory');
}
$('memory-toggle').onclick=()=>{
  memory.collapsed=!memory.collapsed;updateMemoryDock();
  if(!memory.collapsed)for(const array of $('memory-arrays').querySelectorAll('.memory-array'))array.scrollLeft=array.scrollWidth;
  applyCamera();
};
function clearMemory(){memory.selection=null;$('memory-links').replaceChildren();document.querySelectorAll('.memory-selected').forEach(n=>n.classList.remove('memory-selected'));if(!state.selected)$('clear-focus').hidden=true;}
function renderMemory(){
  const t=state.trace;if(!t?.steps.length)return;
  $('memory-links').hidden=false;
  updateMemoryDock();
  $('memory-panel').style.width=memory.width+'px';$('memory-resize').style.right=(memory.width+5)+'px';
  $('memory-context').textContent=`Layer ${t.layer} · Head ${t.head} · KV ${t.kv_head} · FP16`;
  const m=t.steps.at(-1).matrices,prefix=t.steps[0].count,n=t.tokens.length,previousCount=memory.count,added=n>previousCount;memory.count=n;memory.hoverToken=null;
  const dims=m.kc.length+m.vc.length;$('memory-total').textContent=`This KV head · ${memoryBytes(dims*n*2)}`;
  const oldScroll=$('memory-arrays').querySelector('.memory-array')?.scrollLeft||0;$('memory-arrays').replaceChildren();
  for(const [kind,values,color] of [['K',m.kc,colors.k],['V',m.vc,colors.v]]){
    const rows=values.length,tokens=state.timeline.columns[0].blocks.q.plan;
    const base={...KavinProxy.plan(rows,prefix),colIndices:tokens.colIndices,colGap:tokens.colGap,edgeCols:tokens.edgeCols},p=KavinProxy.extendPlan(base,rows,n);
    const section=text('section','memory-section'),heading=text('h4','',`${kind} · ${memoryBytes(rows*n*2)}`),calculation=text('div','memory-size-calculation',`${rows} × ${n} × 2 bytes = ${rows*n*2} B`),wrap=text('div','memory-array'),table=text('table');section.append(heading,calculation,wrap);wrap.append(table);$('memory-arrays').append(section);
    for(let r=-1;r<p.rowIndices.length;r++){
      const tr=text('tr');table.append(tr);
      const rowIndex=text('th','memory-row-index',r<0?'':p.rowIndices[r]===null?'…':String(p.rowIndices[r]+1));
      rowIndex.scope=r<0?'col':'row';tr.append(rowIndex);
      for(const [c,token] of p.colIndices.entries()){
        const td=text(r<0?'th':'td'),button=text('button');button.type='button';td.append(button);tr.append(td);
        if(token!==null){
          td.dataset.token=token;
          if(state.animateAppend&&added&&token>=previousCount)td.classList.add('memory-new-column');
        }
        const item=r<0?null:KavinProxy.cellAt(p,r,c);
        if(item?.omitted){
          button.disabled=true;button.className='memory-omitted';button.style.opacity='1';
          button.title=omittedInfo(item);button.setAttribute('aria-label',button.title);continue;
        }
        if(token===null){button.textContent='…';button.disabled=true;button.className='memory-omitted';continue;}
        const feature=r>=0&&r<p.rowIndices.length?p.rowIndices[r]:null;
        button.dataset.kind=kind;button.dataset.token=token;button.dataset.feature=feature===null?'all':feature;
        if(token===prefix)button.classList.add('memory-decode');
        if(r<0){button.textContent=String(token+1);td.scope='col';}
        else{const value=values[feature][token];button.textContent='';button.style.background=shade(value,kind==='K'?'k':'v');button.style.opacity=String(item.fade);}
        button.title=`Column ${token+1} · ${JSON.stringify(t.tokens[token]?.text??'')}\n${kind}${feature===null?'':` · feature ${feature+1} · value ${values[feature][token]}`}`;
        button.setAttribute('aria-label',button.title);
        button.onclick=()=>{
          const next={kind,token,feature};
          if(memory.selection&&JSON.stringify(memory.selection)===JSON.stringify(next)){clearMemory();return;}
          clearMemory();memory.selection=next;$('clear-focus').hidden=false;button.classList.add('memory-selected');
          const source=memorySource();if(source)focusMatrix(source.block);drawMemoryLinks();
        };
      }
    }
    wrap.addEventListener('scroll',()=>syncMemoryScroll(wrap));
    wrap.onpointermove=event=>{const cell=event.target.closest('[data-token]');highlightMemoryColumn(cell?Number(cell.dataset.token):null);};
    wrap.onpointerleave=()=>highlightMemoryColumn(null);
    wrap.onfocusin=event=>{const cell=event.target.closest('[data-token]');highlightMemoryColumn(cell?Number(cell.dataset.token):null);};
    wrap.onfocusout=event=>{if(!wrap.contains(event.relatedTarget))highlightMemoryColumn(null);};
  }
  for(const array of $('memory-arrays').querySelectorAll('.memory-array'))array.scrollLeft=added?array.scrollWidth:oldScroll;
  drawMemoryLinks();positionCameraToolbar();
}
function memorySource(){
  if(!memory.selection||!state.timeline)return null;
  const {kind,token,feature}=memory.selection,compact=$('compact').checked;
  const index=compact?0:state.trace.steps.findIndex(s=>token>=s.start&&token<s.start+s.count),key=kind==='K'?'kt':'v_operand';
  const block=state.blocks.get(`${index}:${key}`);if(!block)return null;
  const dim=kind==='K'?block.plan.cols:block.plan.rows;
  const range=kind==='K'?{r0:token,r1:token,c0:feature??0,c1:feature??dim-1}:{r0:feature??0,r1:feature??dim-1,c0:token,c1:token};
  return {block,range};
}
function drawMemoryLinks(){
  const svg=$('memory-links');svg.replaceChildren();
  $('memory-panel').querySelectorAll('.memory-selected').forEach(n=>n.classList.remove('memory-selected'));
  if(memory.collapsed)return;
  const source=memorySource();if(!source)return;
  const {kind,token,feature}=memory.selection,{block,range}=source,rect=block.node.getBoundingClientRect();
  const cells=[...$('memory-panel').querySelectorAll(`button[data-kind="${kind}"][data-token="${token}"]`)].filter(b=>feature===null?b.dataset.feature!=='all':Number(b.dataset.feature)===feature);
  for(const target of cells){
    const b=target.getBoundingClientRect(),clip=target.closest('.memory-array').getBoundingClientRect();if(b.left<clip.left-1||b.right>clip.right+1)continue;target.classList.add('memory-selected');
    const f=Number(target.dataset.feature),part=kind==='K'?{...range,c0:f,c1:f}:{...range,r0:f,r1:f};
    for(const region of KavinProxy.regions(block.plan,part)){
      const x=rect.left+(region.x+region.w)*state.zoom,y=rect.top+(region.y+region.h/2)*state.zoom,endX=b.left,endY=b.top+b.height/2;
      svg.append(svgElement('path',{d:`M${x},${y}C${x+40},${y} ${endX-40},${endY} ${endX},${endY}`,fill:'none',stroke:'#238b8b','stroke-width':1.2,'stroke-dasharray':region.hidden?'4 3':''}));
      svg.append(svgElement('rect',{x:rect.left+region.x*state.zoom,y:rect.top+region.y*state.zoom,width:region.w*state.zoom,height:region.h*state.zoom,fill:'none',stroke:'#238b8b','stroke-width':1.2}));
    }
  }
}
$('clear-focus').onclick=()=>{clearMemory();closeInspector();};
let memoryDrag=null;
$('memory-resize').onpointerdown=event=>{event.preventDefault();event.stopPropagation();memoryDrag={x:event.clientX,width:memory.width};$('memory-resize').setPointerCapture(event.pointerId);};
$('memory-resize').onpointermove=event=>{if(!memoryDrag)return;memory.width=Math.max(200,Math.min(innerWidth-300,memoryDrag.width+memoryDrag.x-event.clientX));$('memory-panel').style.width=memory.width+'px';$('memory-resize').style.right=(memory.width+5)+'px';drawMemoryLinks();positionCameraToolbar();};
$('memory-resize').onpointerup=$('memory-resize').onpointercancel=()=>{memoryDrag=null;};
$('memory-resize').onkeydown=event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();memory.width=Math.max(200,Math.min(innerWidth-300,memory.width+(event.key==='ArrowLeft'?20:-20)));renderMemory();};
$('top').onclick=()=>{const b=state.timeline?.columns[0].blocks.x;if(b)focusMatrix(b);};
$('bottom').onclick=()=>{const b=state.timeline?.columns.at(-1).blocks.va;if(b)focusMatrix(b);};
function selectMatrix(block,{names=false}={}){
  state.nameSelection=names;
  $('scene').classList.toggle('name-selection',names);
  document.querySelectorAll('.name-connected').forEach(node=>node.classList.remove('name-connected'));
  if(names){state.selectedValue=null;state.selectedRegion=null;}
  const previousRegion=state.selected===`${block.index}:${block.key}`?state.selectedRegion:null;
  state.selectedRegion=null;
  document.querySelectorAll('.is-selected').forEach(node=>node.classList.remove('is-selected','has-cell-focus'));block.node.classList.add('is-selected');
  state.selected=`${block.index}:${block.key}`;const p=block.plan;
  $('inspector').hidden=true;$('inspect-title').textContent=descriptions[block.kind][0];$('inspect-description').textContent=descriptions[block.kind][1];
  const context=block.parameter?`${state.trace.weights[block.key].storage_type} GGUF · head ${state.trace.weights[block.key].head}`:block.shared?`shared cache · through token ${state.trace.steps[block.sourceStep].start+state.trace.steps[block.sourceStep].count}`:block.index?`decode ${block.index}`:'prefill';
  $('inspect-shape').textContent=`${p.rows} × ${p.cols} · ${p.retained} actual entries${p.rowGap||p.colGap?' · corner proxy':''} · ${context}${block.appendOnly?' · appended entries only':''}`;
  const selected=state.selectedValue;
  if(selected?.key===state.selected&&p.rowIndices.includes(selected.row)&&p.colIndices.includes(selected.col))inspectCell(block,KavinProxy.cellAt(p,p.rowIndices.indexOf(selected.row),p.colIndices.indexOf(selected.col)));
  else if(previousRegion?.omitted)inspectCell(block,previousRegion);
  else{state.selectedValue=null;$('inspect-value').hidden=true;}
  if(names){$('clear-focus').hidden=false;drawNameConnections(block);layoutMatrixNames();drawNameConnections(block);}
  else{drawConnections(block.index,block.key,state.selectedRegion);layoutMatrixNames();}
}
function closeInspector(){clearSelectionColors();clearMemory();state.nameSelection=false;$('scene').classList.remove('name-selection');document.querySelectorAll('.name-connected').forEach(n=>n.classList.remove('name-connected'));$('clear-focus').hidden=true;state.selected=null;state.selectedValue=null;state.selectedRegion=null;$('inspector').hidden=true;$('connections').replaceChildren();document.querySelectorAll('.is-selected').forEach(n=>n.classList.remove('is-selected','has-cell-focus'));layoutMatrixNames();updateMatrixCursor();}
$('close-inspector').onclick=closeInspector;
$('focus-matrix').onclick=()=>focusMatrix(state.blocks.get(state.selected));
function focusMatrix(block,requestedZoom){
  if(!block)return;const r=$('viewport').getBoundingClientRect();
  const rail=Math.min(110,r.width*.3);
  const zoom=requestedZoom??Math.max(.1,Math.min(1.5,(r.width-rail-memoryWidth()-80)/block.w,(r.height-95)/block.h));
  if(requestedZoom!==undefined){animateCamera({zoom,x:(rail+r.width-memoryWidth()-75)/2-(block.x+block.w/2)*zoom,y:(80+r.height-24)/2-(block.y+block.h/2)*zoom});return;}
  animateCamera({zoom,x:rail+5-block.x*zoom,y:85-block.y*zoom});
}
function markFocusedNumbers(block){
  for(const value of block.node.querySelectorAll('.proxy-value')){
    const x=Number(value.getAttribute('x')),y=Number(value.getAttribute('y'));
    value.classList.toggle('focus-number',Boolean(block.focusRects?.some(r=>!r.hidden&&x>=r.x&&x<=r.x+r.w&&y>=r.y&&y<=r.y+r.h)));
    value.classList.toggle('selected-number',Boolean(value.classList.contains('focus-number')&&state.selected===`${block.index}:${block.key}`&&state.selectedRegion&&!state.selectedRegion.omitted&&Number(value.dataset.row)===state.selectedRegion.row&&Number(value.dataset.col)===state.selectedRegion.col));
    const maxWidth=block.plan.cellWidth-3/state.zoom,fontSize=value.classList.contains('focus-number')?Math.max(11,9/state.zoom):11;
    if(value.textContent.length*fontSize*.61>maxWidth){value.setAttribute('textLength',maxWidth);value.setAttribute('lengthAdjust','spacingAndGlyphs');}
    else{value.removeAttribute('textLength');value.removeAttribute('lengthAdjust');}
  }
}
function clearSelectionColors(){
  $('scene').classList.remove('selection-color-focus');
  document.querySelectorAll('.matrix.name-associated').forEach(node=>node.classList.remove('name-associated'));
  document.querySelectorAll('.focus-colors').forEach(canvas=>canvas.remove());
  for(const block of state.blocks.values()){delete block.focusRects;markFocusedNumbers(block);}
  document.querySelectorAll('.focus-number').forEach(value=>value.classList.remove('focus-number'));
}
function focusSelectionColors(source,selected,math){
  clearSelectionColors();$('scene').classList.add('selection-color-focus');
  const ranges=new Map([[`${source.index}:${source.key}`,selected.slice()]]);
  for(const operand of math.operands){
    const id=`${operand.index}:${operand.key}`,block=state.blocks.get(id);
    if(block)ranges.set(id,[...(ranges.get(id)||[]),...KavinProxy.regions(block.plan,operand.range)]);
  }
  for(const [id,rects] of ranges){
    const block=state.blocks.get(id),base=block.node.querySelector('canvas'),canvas=document.createElement('canvas');
    block.focusRects=rects;markFocusedNumbers(block);
    canvas.className='focus-colors';canvas.setAttribute('aria-hidden','true');canvas.width=base.width;canvas.height=base.height;
    canvas.style.width=block.w+'px';canvas.style.height=block.h+'px';
    const ctx=canvas.getContext('2d');ctx.scale(2,2);ctx.beginPath();
    for(const rect of rects)if(!rect.hidden)ctx.rect(rect.x,rect.y,rect.w,rect.h);
    ctx.clip();ctx.drawImage(base,0,0,block.w,block.h);block.node.append(canvas);
  }
}
function drawRangeHighlights(svg,block,rects,selected=false){
  for(const rect of rects){
    const classes='math-range'+(selected?' math-selected':'')+(rect.hidden?' hidden-range':'');
    svg.append(svgElement('rect',{x:block.x+rect.x,y:block.y+rect.y,width:rect.w,height:rect.h,class:classes+' range-fill'}));
    svg.append(svgElement('path',{d:KavinProxy.rangeOutline(rect,rects),transform:`translate(${block.x} ${block.y})`,class:classes+' range-outline','vector-effect':'non-scaling-stroke'}));
  }
}
function drawNameConnections(block){
  clearSelectionColors();
  const svg=$('connections');svg.replaceChildren();$('inspect-math').replaceChildren();
  const p=block.plan,math=KavinProxy.mathLinks(state.trace,state.timeline,block.index,block.key,KavinProxy.matrixRange(p));
  if(math.formula)$('inspect-math').append(text('code','',math.formula));
  $('inspect-math').append(text('small','',math.note));
  const viewport=$('viewport').getBoundingClientRect();
  const box=b=>{const r=b.nameNode.getBoundingClientRect();return{x:(r.left-viewport.left-state.x)/state.zoom,y:(r.top-viewport.top-state.y)/state.zoom,w:r.width/state.zoom,h:r.height/state.zoom};};
  const nameBoxes=new Map([...state.blocks.values()].filter(b=>b.nameNode).map(b=>[b,box(b)]));
  block.nameNode.classList.add('name-connected');
  block.node.classList.add('name-associated');
  const seen=new Set();
  for(const operand of math.operands){
    const id=`${operand.index}:${operand.key}`,target=state.blocks.get(id);if(!target?.nameNode||seen.has(id))continue;seen.add(id);
    target.nameNode.classList.add('name-connected');
    target.node.classList.add('name-associated');
    const a=nameBoxes.get(block),b=nameBoxes.get(target),{start,end,vertical}=KavinProxy.connectionAnchors(a,b,a,b);
    const padding=5/state.zoom,obstacles=[...nameBoxes].filter(([other])=>other!==block&&other!==target).map(([,r])=>({x:r.x-padding,y:r.y-padding,w:r.w+2*padding,h:r.h+2*padding}));
    const dx=end.x-start.x,dy=end.y-start.y,steps=Math.max(80,Math.ceil(Math.hypot(dx,dy)*state.zoom/3));
    let d='';
    // Bow away from aligned names instead of degenerating into a straight line.
    for(const bend of [24,-24,48,-48,80,-80,120,-120,180,-180,260,-260,384,-384]){
      const offset=bend/state.zoom;
      const c1=vertical?{x:start.x+offset,y:start.y+dy/3}:{x:start.x+dx/3,y:start.y+offset};
      const c2=vertical?{x:end.x+offset,y:end.y-dy/3}:{x:end.x-dx/3,y:end.y+offset};
      let blocked=false;
      for(let i=1;i<steps&&!blocked;i++){
        const t=i/steps,u=1-t,x=u*u*u*start.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*end.x,y=u*u*u*start.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*end.y;
        blocked=obstacles.some(r=>x>r.x&&x<r.x+r.w&&y>r.y&&y<r.y+r.h);
      }
      if(!blocked){d=`M${start.x},${start.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${end.x},${end.y}`;break;}
    }
    if(!d){const points=KavinProxy.routeConnection(start,end,obstacles);if(!points)continue;d=KavinProxy.curvedConnection(points,obstacles,32/state.zoom);}
    const path=svgElement('path',{d,class:'math-name-link','data-source':`${block.index}:${block.key}`,'data-target':id});
    path.append(svgElement('title',{},operand.role));svg.append(path);
  }
}
function drawConnections(index,key,item){
  const cellFocused=Boolean(item);clearSelectionColors();
  const svg=$('connections');svg.replaceChildren();$('inspect-math').replaceChildren();
  const source=state.blocks.get(`${index}:${key}`);if(!source)return;
  const p=source.plan;
  item=item||KavinProxy.matrixRange(p);
  const math=KavinProxy.mathLinks(state.trace,state.timeline,index,key,item);
  if(math.formula)$('inspect-math').append(text('code','',math.formula));
  $('inspect-math').append(text('small','',math.note));
  const selection={r0:item.rowRange[0],r1:item.rowRange[1],c0:item.colRange[0],c1:item.colRange[1]};
  const selected=KavinProxy.regions(p,selection);
  if(cellFocused)focusSelectionColors(source,selected,math);
  drawRangeHighlights(svg,source,selected,true);
  if(!selected.length)return;
  const left=Math.min(...selected.map(r=>r.x)),right=Math.max(...selected.map(r=>r.x+r.w)),top=Math.min(...selected.map(r=>r.y)),bottom=Math.max(...selected.map(r=>r.y+r.h));
  const body=block=>({x:block.x+block.plan.padX,y:block.y+block.plan.padY,w:block.w-block.plan.padX,h:block.h-block.plan.padY});
  const selectedBounds={x:source.x+left,y:source.y+top,w:right-left,h:bottom-top};
  const ribbonDefs=svgElement('defs');svg.append(ribbonDefs);let ribbonIndex=0;
  const shadow=svgElement('filter',{id:'math-tape-shadow',x:'-30%',y:'-30%',width:'160%',height:'180%','color-interpolation-filters':'sRGB'});
  shadow.append(svgElement('feDropShadow',{dx:0,dy:5,stdDeviation:3,'flood-color':'#243655','flood-opacity':.16}));ribbonDefs.append(shadow);
  for(const operand of math.operands){
    const target=state.blocks.get(`${operand.index}:${operand.key}`),regions=KavinProxy.mergeRegions(KavinProxy.regions(target.plan,operand.range));
    if(!regions.length)continue;
    drawRangeHighlights(svg,target,regions);
    // One ribbon per logical operand, rather than overlapping ribbons for
    // every retained proxy slice. Highlights still distinguish hidden spans.
    const x=Math.min(...regions.map(r=>r.x)),y=Math.min(...regions.map(r=>r.y));
    const w=Math.max(...regions.map(r=>r.x+r.w))-x,h=Math.max(...regions.map(r=>r.y+r.h))-y;
    const range={x:target.x+x,y:target.y+y,w,h},hidden=regions.every(r=>r.hidden),hasHidden=regions.some(r=>r.hidden);
    const {start,end,vertical,sourceWidth,targetWidth}=KavinProxy.connectionAnchors(body(source),body(target),selectedBounds,range);
    const detail=operand.role+(hasHidden?' · includes omitted cells':'');
    {
      const id=`math-ribbon-${ribbonIndex++}`,gradient=svgElement('linearGradient',{id,gradientUnits:'userSpaceOnUse',x1:start.x,y1:start.y,x2:end.x,y2:end.y});
      const first=colors[source.kind],last=colors[target.kind],shine=first.map((v,i)=>Math.round((v+last[i]+255)/3));
      for(const [offset,color,opacity] of [[0,first,.18],[.45,shine,.24],[1,last,.18]])gradient.append(svgElement('stop',{offset,'stop-color':`rgb(${color.join(',')})`,'stop-opacity':opacity}));
      ribbonDefs.append(gradient);
      const lift=Math.max(24,Math.min(72,Math.hypot(end.x-start.x,end.y-start.y)*.14));
      const ribbon=svgElement('path',{d:KavinProxy.ribbonConnection(start,end,sourceWidth,targetWidth,vertical,lift),class:'math-ribbon'+(hidden?' hidden-ribbon':''),fill:`url(#${id})`,filter:'url(#math-tape-shadow)','stroke-width':.75/state.zoom,'vector-effect':'none','data-source':`${index}:${key}`,'data-target':`${operand.index}:${operand.key}`});
      ribbon.append(svgElement('title',{},detail));svg.append(ribbon);
    }
  }
}
let cameraFrame=0,cameraTarget=null;
function layoutMatrixNames(){
  const entries=[...state.blocks.values()].filter(b=>b.nameNode).map(b=>{
    const node=b.nameNode;
    node.textContent=node.dataset.fullName==='Scores'&&state.zoom<1?'S':node.dataset.fullName;
    node.style.visibility='';node.style.marginTop='0px';
    return {node,priority:node.classList.contains('name-connected')?3:state.selected===`${b.index}:${b.key}`?2:b.index===0?1:0};
  }).sort((a,b)=>b.priority-a.priority);
  const placed=[],overlaps=r=>placed.some(p=>r.left<p.right+3&&r.right>p.left-3&&r.top<p.bottom+3&&r.bottom>p.top-3);
  for(const entry of entries){
    let r=entry.node.getBoundingClientRect();
    if(overlaps(r)&&entry.priority>=2){
      for(const offset of Array.from({length:16},(_,i)=>(i%2?1:-1)*(Math.floor(i/2)+1)*18)){
        entry.node.style.marginTop=offset/state.zoom+'px';r=entry.node.getBoundingClientRect();if(!overlaps(r))break;
      }
    }
    if(overlaps(r)&&entry.priority<2)entry.node.style.visibility='hidden';
    else placed.push(r);
  }
}
function stopCamera(){cancelAnimationFrame(cameraFrame);cameraFrame=0;cameraTarget=null;$('viewport').dataset.animating='false';}
function animateCamera(target){
  stopCamera();cameraTarget=target;
  const start={zoom:state.zoom,x:state.x,y:state.y},started=performance.now();
  const duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:340;
  $('viewport').dataset.animating='true';
  function frame(now){
    const t=duration?Math.min(1,(now-started)/duration):1,eased=cameraEase(t);
    for(const key of ['zoom','x','y'])state[key]=start[key]+(target[key]-start[key])*eased;
    applyCamera();
    if(t<1)cameraFrame=requestAnimationFrame(frame);else stopCamera();
  }
  cameraFrame=requestAnimationFrame(frame);
}
function drawDiagramBrackets(r){
  if(!state.timeline||!state.blocks.size)return;
  const blocks=[...state.blocks.values()],z=state.zoom;
  const left=Math.min(...blocks.map(b=>b.x+b.plan.padX)),right=Math.max(...blocks.map(b=>b.x+b.w));
  const bodyLeft=state.x+left*z,bodyRight=state.x+right*z;
  const outerLeft=state.x+Math.min(...blocks.map(b=>b.x))*z;
  const firstTop=state.y+state.timeline.y.x*z,diagramBottom=state.y+state.timeline.height*z;
  const headerTop=Math.max(8,firstTop-52),decodeHeaders=[];
  const innerLeft=Math.max(8,outerLeft-92),contentLeft=innerLeft+92,contentTop=headerTop+51;
  // Clip in screen coordinates: growing shared caches can extend left of scene x=0.
  const clip=$('scene-clip');
  clip.style.clipPath=`polygon(${contentLeft}px ${contentTop}px,${r.width}px ${contentTop}px,${r.width}px ${r.height}px,${contentLeft}px ${r.height}px)`;
  const fades=[];
  if(outerLeft<contentLeft-.5)fades.push(`linear-gradient(to right,transparent ${contentLeft}px,black ${contentLeft+18}px)`);
  if(firstTop<contentTop-.5)fades.push(`linear-gradient(to bottom,transparent ${contentTop}px,black ${contentTop+14}px)`);
  clip.style.maskImage=fades.join(',')||'none';
  clip.style.maskComposite='intersect';
  for(const col of state.timeline.columns){
    const header=$('step-labels').children[col.index];if(!header)continue;
    const gap=Math.min(8,col.width*z*.08),x=Math.max(contentLeft,state.x+col.x*z),end=Math.min(r.width-12,state.x+(col.x+col.width)*z-gap);
    header.hidden=end-x<2||diagramBottom<8||firstTop>r.height;
    position(header,x,headerTop,Math.max(0,end-x),43);header.classList.toggle('narrow',end-x<95);
    header.querySelector('strong').textContent=col.index&&end-x<60?header.dataset.short:header.dataset.label;header.querySelector('strong').hidden=false;
    if(col.index&&!header.hidden)decodeHeaders.push({header,left:x,right:end,index:col.index});
  }
  const grouped=decodeHeaders.length>1&&decodeHeaders.some(h=>h.right-h.left<80);
  $('decode-group').hidden=!grouped;
  if(grouped){
    decodeHeaders.forEach(h=>h.header.querySelector('strong').hidden=true);
    const first=decodeHeaders[0],last=decodeHeaders.at(-1);$('decode-group').textContent=`Decode ${first.index}–${last.index}`;position($('decode-group'),first.left,headerTop+11,last.right-first.left,20);
  }
  const horizontalVisible=bodyRight>8&&bodyLeft<r.width-8;
  function placeBracket(node,x,width){
    const naturalTop=state.y+Number(node.dataset.top)*z,naturalBottom=naturalTop+Number(node.dataset.height)*z;
    const top=Math.max(contentTop,naturalTop),height=Math.max(0,Math.min(r.height-8,naturalBottom)-top);
    node.hidden=!horizontalVisible||height<1;
    position(node,x,top,width,height);
    const title=node.querySelector('.vertical-title,span');
    title.hidden=height<18;
    title.style.top='3px';
  }
  for(const label of $('axis-labels').querySelectorAll('.row-label')){
    const top=Number(label.dataset.top),bottom=top+Number(label.dataset.height);
    const members=blocks.filter(b=>b.y>=top&&b.y<bottom);
    const groupLeft=members.length?state.x+Math.min(...members.map(b=>b.x+b.plan.padX))*z:bodyLeft;
    const groupRight=members.length?state.x+Math.max(...members.map(b=>b.x+b.w))*z:bodyRight;
    placeBracket(label,innerLeft,88);
    label.hidden=label.hidden||groupRight<8||groupLeft>r.width-8;
  }

}
function applyCamera(){
  constrainCamera();
  $('viewport').scrollTop=0;$('viewport').scrollLeft=0;
  applySceneTransform($('scene'),state);
  updateMatrixCursor();
  layoutMatrixNames();
  if(state.nameSelection&&state.blocks.has(state.selected))drawNameConnections(state.blocks.get(state.selected));
  $('zoom').value=Math.round(state.zoom*100)+'%';
  $('zoom-in').title=`Zoom in (+) · ${$('zoom').value}`;$('zoom-out').title=`Zoom out (−) · ${$('zoom').value}`;
  const r=$('viewport').getBoundingClientRect();
  drawDiagramBrackets(r);
  for(const block of state.blocks.values()){
    const x=state.x+block.x*state.zoom,y=state.y+block.y*state.zoom;
    const visible=x+block.w*state.zoom>-60&&x<r.width+60&&y+block.h*state.zoom>-60&&y<r.height+60;
    drawAnnotations(block,visible&&state.zoom>=.7-1e-6?$('numbers').checked?'values':'axes':'');
    // CSS camera transforms also scale SVG viewports; compensate explicitly
    // so the outer contour remains one screen pixel at every zoom level.
    for(const line of block.node.querySelectorAll('.matrix-outline,.append-boundary')){
      line.setAttribute('stroke-width',1/state.zoom);
      if(line.classList.contains('omitted-outline'))line.setAttribute('stroke-dasharray',`${3/state.zoom} ${3/state.zoom}`);
    }
  }
  for(const tape of $('connections').querySelectorAll('.math-ribbon'))tape.setAttribute('stroke-width',.75/state.zoom);
  drawMemoryLinks();positionCameraToolbar();
  $('scale-note').textContent=$('numbers').checked?'Original corner cells · shaded gaps omitted':'Corner proxies · numbers hidden';
}
function cameraGeometry(){
  if(!state.blocks.size)return null;
  const blocks=[...state.blocks.values()],r=$('viewport').getBoundingClientRect();
  return {
    left:Math.min(...blocks.map(b=>b.x+b.plan.padX)),right:Math.max(...blocks.map(b=>b.x+b.w)),
    top:Math.min(...blocks.map(b=>b.y+b.plan.padY)),bottom:Math.max(...blocks.map(b=>b.y+b.h)),
    viewport:{left:Math.min(110,r.width*.3)+10,right:Math.max(Math.min(110,r.width*.3)+40,r.width-memoryWidth()-76),top:80,bottom:Math.max(120,r.height-12)}
  };
}
function constrainCamera(){
  const b=cameraGeometry();if(!b)return;
  const v=b.viewport,z=state.zoom;
  const visibleX=Math.min(96,(v.right-v.left)/2,(b.right-b.left)*z);
  const visibleY=Math.min(64,(v.bottom-v.top)/2,(b.bottom-b.top)*z);
  state.x=Math.max(v.left+visibleX-b.right*z,Math.min(v.right-visibleX-b.left*z,state.x));
  state.y=Math.max(v.top+visibleY-b.bottom*z,Math.min(v.bottom-visibleY-b.top*z,state.y));
}
function overview({animate=true}={}){
  const b=cameraGeometry();if(!b)return;const v=b.viewport;
  const zoom=Math.max(.025,Math.min(1,(v.right-v.left-20)/(b.right-b.left),(v.bottom-v.top-20)/(b.bottom-b.top)));
  const target={zoom,x:(v.left+v.right)/2-(b.left+b.right)*zoom/2,y:(v.top+v.bottom)/2-(b.top+b.bottom)*zoom/2};
  if(animate)animateCamera(target);else{stopCamera();Object.assign(state,target);applyCamera();}
}
function recenter(){
  const b=cameraGeometry();if(!b)return;const v=b.viewport;
  animateCamera({zoom:state.zoom,x:(v.left+v.right)/2-(b.left+b.right)*state.zoom/2,y:(v.top+v.bottom)/2-(b.top+b.bottom)*state.zoom/2});
}
function zoomBy(factor,cx,cy){
  const r=$('viewport').getBoundingClientRect();cx=cx??r.width/2;cy=cy??r.height/2;
  const next=Math.max(.025,Math.min(3,(cameraTarget?.zoom||state.zoom)*factor));
  const [sceneX,sceneY]=cameraTransform(state).invert([cx,cy]);
  animateCamera({zoom:next,x:cx-sceneX*next,y:cy-sceneY*next});
}
$('overview').onclick=overview;$('recenter').onclick=recenter;$('zoom-in').onclick=()=>zoomBy(1.25);$('zoom-out').onclick=()=>zoomBy(.8);
let drag,spacePan=false,panMoved=false;
$('viewport').addEventListener('pointerdown',event=>{
  const matrix=event.target.closest('.matrix,.scene-matrix-name');
  if(!state.trace||event.button>1||event.target.closest('button')&&!matrix&&!spacePan&&event.button!==1)return;
  stopCamera();panMoved=false;event.preventDefault();drag={x:event.clientX,y:event.clientY,ox:state.x,oy:state.y};
  document.querySelectorAll('.cell-hover').forEach(node=>node.hidden=true);
  (matrix||$('viewport')).setPointerCapture(event.pointerId);
});
$('viewport').addEventListener('pointermove',event=>{if(!drag)return;if(Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>3)panMoved=true;if(!panMoved)return;$('viewport').classList.add('dragging');state.x=drag.ox+event.clientX-drag.x;state.y=drag.oy+event.clientY-drag.y;applyCamera();});
for(const event of ['pointerup','pointercancel'])$('viewport').addEventListener(event,()=>{drag=null;$('viewport').classList.remove('dragging');});
$('viewport').addEventListener('wheel',event=>{if(!state.trace)return;event.preventDefault();const r=$('viewport').getBoundingClientRect();if(event.ctrlKey||event.metaKey)zoomBy(Math.exp(-event.deltaY*.004),event.clientX-r.left,event.clientY-r.top);else{stopCamera();state.x-=event.shiftKey?event.deltaY:event.deltaX;state.y-=event.shiftKey?0:event.deltaY;applyCamera();}},{passive:false});
$('viewport').addEventListener('keydown',event=>{
  if(event.key===' '){spacePan=true;event.preventDefault();}
  if(event.target!==$('viewport'))return;
  const moves={ArrowLeft:[30,0],ArrowRight:[-30,0],ArrowUp:[0,30],ArrowDown:[0,-30]};
  if(moves[event.key]){stopCamera();event.preventDefault();state.x+=moves[event.key][0];state.y+=moves[event.key][1];applyCamera();}
  if(event.key==='0')overview();if(event.key==='+')zoomBy(1.25);if(event.key==='-')zoomBy(.8);
});
window.addEventListener('resize',()=>{if(state.trace)applyCamera();});
window.addEventListener('keyup',event=>{if(event.key===' ')spacePan=false;});window.addEventListener('blur',()=>{spacePan=false;});
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeInspector();});
actions.exportTrace=()=>{if(!state.trace)return;const blob=new Blob([JSON.stringify(state.trace)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='kevin-trace.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
async function init(){
  try{
    let status=await getStatus();while(['loading','validating','downloading'].includes(status.phase)){await new Promise(r=>setTimeout(r,600));status=await getStatus();}
    if(status.phase==='error')toast(status.message);
    if(!status.engine_ready)toast('Build the inference engine with python3 scripts/setup.py, then restart KeVin.');
    if(status.phase==='ready'){
      const response=DEMO?{ok:true,json:()=>api('/api/trace')}:await fetch('/api/trace');if(response.ok){state.trace=await response.json();if(state.trace.prompt!==undefined)$('prompt').value=state.trace.prompt;render();overview({animate:false});refreshControls();}
    }
  }catch(error){toast(error.message);}
}
init();

}

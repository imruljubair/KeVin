// Formula descriptions contain only observed values and model-provided metadata.
// No inference is reconstructed from the rounded corner proxies.
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const mi=s=>`<mi mathvariant="normal">${esc(s)}</mi>`,mn=s=>`<mn>${esc(s)}</mn>`,mo=s=>s==='·'?'<mo class="math-product" lspace="0.25em" rspace="0.25em">·</mo>':`<mo>${esc(s)}</mo>`;
const word=s=>`<mtext>${esc(s)}</mtext>`;
const row=(...s)=>`<mrow>${s.join('')}</mrow>`;
const frac=(a,b)=>`<mfrac>${row(a)}${row(b)}</mfrac>`;
const dot=(a,b)=>row(mi(a),mo('·'),mi(b));
const power=(base,exponent)=>`<msup>${row(base)}${row(exponent)}</msup>`;
const call=(name,arg)=>row(mi(name),mo('('),arg,mo(')'));
const eq=(a,b)=>row(a,mo('='),b);
const math=s=>`<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">${s}</math>`;
const range=(r0,r1,c0,c1)=>({r0,r1,c0,c1});
export function ropePair(row,dimensions,type){
  if(row>=dimensions||!Number.isInteger(dimensions)||dimensions<=0||dimensions%2||![0,2].includes(type))return null;
  const first=type===2?row%(dimensions/2):row-row%2;
  const second=type===2?first+dimensions/2:first+1;
  return {first,second,pair:type===2?first:first/2,secondSelected:row===second};
}
export function ropeAngle(config,pair,position){
  if(!config||![0,2].includes(config.type)||config.extension!==0||config.magnitude!==1||!(config.base>0)||!(config.scale>0))return null;
  const factor=config.factors?.[pair]??1;
  if(!(factor>0))return null;
  const exponent=2*pair/config.dimensions;
  const frequency=Math.pow(config.base,-exponent)*config.scale/factor;
  return {exponent,factor,frequency,angle:position*frequency};
}
function rotationTerm(second){
  const cos=call('cos',word('angle')),sin=call('sin',word('angle'));
  const coefficients=second?[sin,cos]:[cos,row(mo('−'),sin)];
  const matrix=rows=>row(mo('['),`<mtable columnspacing="0.6em" rowspacing="0.15em">${rows.map(cells=>`<mtr>${cells.map(cell=>`<mtd>${cell}</mtd>`).join('')}</mtr>`).join('')}</mtable>`,mo(']'));
  return row(matrix([coefficients]),mo('·'),matrix([[mi('u')],[mi('v')]]));
}
function rotationCases(){
  return row(mo('{'),`<mtable columnalign="left left"><mtr><mtd>${rotationTerm(false)}</mtd><mtd>${word('first')}</mtd></mtr><mtr><mtd>${rotationTerm(true)}</mtd><mtd>${word('second')}</mtd></mtr></mtable>`);
}
// Keep unavailable settings as named intermediates instead of eliding RoPE.
function symbolicFrequency(config,pair,dimensions){
  const base=config?.base>0?mn(config.base):word('base');
  const scale=config?.scale>0?mn(config.scale):word('scale');
  const knownPair=typeof pair==='number';
  const factors=config?.factors;
  const factor=config?(factors?(knownPair?mn(factors[pair]):factors.every(x=>x===1)?mn(1):word('factor')):mn(1)):word('factor');
  const exponent=frac(row(mn(2),mo('×'),knownPair?mn(pair):mi('i')),dimensions?mn(dimensions):word('dimensions'));
  const baseFrequency=frac(mn(1),row(power(base,exponent),...(factor===mn(1)?[]:[mo('·'),factor])));
  const lines=[math(eq(word('frequency'),row(...(scale===mn(1)?[]:[scale,mo('·')]),baseFrequency)))];
  if(config?.extension){
    lines[0]=math(eq(word('base frequency'),baseFrequency));
    lines.push(math(eq(word('frequency'),row(mo('('),mn(1),mo('−'),word('mix'),mo(')'),mo('·'),scale,mo('·'),word('base frequency'),mo('+'),word('mix'),mo('·'),word('base frequency')))));
  }
  return lines;
}
function rotationGain(config){
  if(!config||config.extension===0&&config.magnitude===1)return [];
  let gain=mn(config.magnitude);
  if(config.extension)gain=row(gain,mo('·'),mo('('),mn(1),mo('+'),mn(.1),mo('·'),call('ln',frac(mn(1),mn(config.scale))),mo(')'));
  return [math(eq(word('gain'),gain))];
}
// A RoPE selection always explains both members of each represented pair.
function pairedRopeExplanation(trace,index,key,item){
  if(!['k','q'].includes(key)||!trace.weights?.['w'+key])return null;
  const step=trace.steps[index],config=step.explanation?.[key],weights=trace.weights;
  const dims=config?.dimensions??weights.rope_dims,type=config?.type??weights.rope_type;
  if(!ropePair(item.rowRange[0],dims,type)||item.rowRange[1]>=dims)return null;
  const pairs=new Map();
  for(let r=item.rowRange[0];r<=item.rowRange[1];r++){
    const pair=ropePair(r,dims,type);pairs.set(pair.pair,pair);
  }
  const all=[...pairs.values()].sort((a,b)=>a.pair-b.pair),cols=weights['w'+key].cols;
  const group=(rows,index,key,label,role,c0,c1)=>{
    const unique=[...new Set(rows)].sort((a,b)=>a-b);
    return {index,key,label,role,range:range(unique[0],unique.at(-1),c0,c1),
      ranges:unique.map(r=>range(r,r,c0,c1)),vector:unique.length>1||c0!==c1};
  };
  const first=all.map(p=>p.first),second=all.map(p=>p.second),[c0,c1]=item.colRange;
  const results=[group(first,index,key,'a','First paired output',c0,c1),group(second,index,key,'b','Second paired output',c0,c1)].map((op,pairSide)=>({...op,result:true,pairSide}));
  const operands=[{...group(first,0,'w'+key,'c','First weight row',0,cols-1),pairSide:0},
    {...group(second,0,'w'+key,'d','Paired weight row',0,cols-1),pairSide:1},
    {index,key:'x',label:'e',role:'Input column',range:range(0,cols-1,c0,c1)}];
  const product=letter=>row(dot(letter,'e'),...(weights.transforms?.[key]?.includes('bias')?[mo('+'),word('bias')]:[]));
  const lines=[math(eq(mi('u'),product('c'))),math(eq(mi('v'),product('d')))];
  const pairIndex=all.length===1?all[0].pair:'i';
  const position=c0===c1?mn(c0):mi('j');
  const details=[math(eq(word('position'),row(mn(step.start),mo('+'),position))),
    math(eq(word('pair'),typeof pairIndex==='number'?mn(pairIndex):mi(pairIndex))),
    ...symbolicFrequency(config,pairIndex,dims),...rotationGain(config),
    math(eq(word('angle'),row(word('position'),mo('·'),word('frequency'))))];
  const matrix=rows=>row(mo('['),`<mtable columnspacing="0.6em" rowspacing="0.15em">${rows.map(cells=>`<mtr>${cells.map(cell=>`<mtd>${cell}</mtd>`).join('')}</mtr>`).join('')}</mtable>`,mo(']'));
  const cos=call('cos',word('angle')),sin=call('sin',word('angle'));
  let rotated=row(matrix([[cos,row(mo('−'),sin)],[sin,cos]]),mo('·'),matrix([[mi('u')],[mi('v')]]));
  if(config&&(config.extension!==0||config.magnitude!==1))rotated=row(word('gain'),mo('·'),rotated);
  if(weights.head_norm)rotated=call('normalize',rotated);
  const final=math(row(matrix([[mi('a')],[mi('b')]]),mo(key==='k'||weights.head_norm?'≈':'='),rotated));
  return {title:'RoPE pair',value:null,result:results[0],results,operands,details,
    lines:[...lines,final],derivation:[...lines,...details,final],
    notes:[`Dot products use all ${cols} components, including the hidden middle.`,...(!config?['This trace does not include RoPE settings; symbolic parameters are shown.']:[]),...(key==='k'?['K is rounded to F16 in the cache.']:[])]};
}
export function cellExplanation(trace,timeline,index,key,item,links){
  if(!item)return null;
  const pairExplanation=pairedRopeExplanation(trace,index,key,item);if(pairExplanation)return pairExplanation;
  if(item.omitted||item.row!==item.rowRange[1]||item.col!==item.colRange[1])return rangeExplanation(trace,index,key,item,links);
  const step=trace.steps[index],weights=trace.weights||{},r=item.row,c=item.col;
  const block=timeline.columns[index].blocks[key];
  const value=block.values?.[r]?.[c];
  const out={title:'Selected cell',value,operands:[],lines:[],notes:[],details:[],result:{index,key,range:range(r,r,c,c)}};
  const add=(index,key,region,label,role)=>out.operands.push({index,key,range:region,label,role});
  const line=s=>out.lines.push(math(s));
  if(['k','q','v'].includes(key)&&weights['w'+key]){
    const config=step.explanation?.[key],rot=config?.dimensions??weights.rope_dims;
    const pair=key==='v'?null:ropePair(r,rot,config?.type??weights.rope_type);
    const bias=(weights.transforms?.[key]||'').includes('bias');
    const norm=Boolean(weights.head_norm&&key!=='v');
    out.title=key.toUpperCase()+' projection'+(pair?' · RoPE':'');
    const cols=weights['w'+key].cols;
    if(!timeline.columns[0].blocks['w'+key])out.notes.push('Expand Weights and K/V to see the labeled weight rows.');
    const product=(a,b)=>bias?row(dot(a,b),mo('+'),word('bias')):dot(a,b);
    if(pair){
      add(0,'w'+key,range(pair.first,pair.first,0,cols-1),'b','First weight row');
      add(0,'w'+key,range(pair.second,pair.second,0,cols-1),'c','Paired weight row');
      add(index,'x',range(0,cols-1,c,c),'d','Input column');
      line(eq(mi('u'),product('b','d')));line(eq(mi('v'),product('c','d')));
      const rotated=rotationTerm(pair.secondSelected);
      const extended=Boolean(config&&!ropeAngle(config,pair.pair,step.start+c));
      const transformed=extended?row(word('gain'),mo('·'),mo('('),rotated,mo(')')):rotated;
      line(row(mi('a'),mo(norm||key==='k'?'≈':'='),norm?call('normalize',transformed):transformed));
      const position=step.start+c,angle=ropeAngle(config,pair.pair,position);
      out.details.push(math(eq(word('position'),row(mn(step.start),mo('+'),mn(c)))));
      out.details.push(math(eq(word('pair'),mn(pair.pair))));
      if(angle){
        const denominator=power(mn(config.base),frac(row(mn(2),mo('×'),mn(pair.pair)),mn(config.dimensions)));
        out.details.push(math(eq(word('frequency'),row(frac(mn(1),denominator),...(config.scale!==1?[mo('×'),mn(config.scale)]:[]),...(angle.factor!==1?[mo('÷'),mn(angle.factor)]:[])))));
        out.details.push(math(eq(word('angle'),row(word('position'),mo('·'),word('frequency')))));
      }else{
        out.details.push(...symbolicFrequency(config,pair.pair,rot),...rotationGain(config));
        out.details.push(math(eq(word('angle'),row(word('position'),mo('·'),word('frequency')))));
        out.notes.push(config?'This model uses an extended RoPE transform; the simple angle calculation does not apply.':'This trace does not include RoPE frequencies or intermediate values. Restart KeVin and prefill again to capture them.');
      }
      // Read like a worked notebook derivation: intermediates, angle setup,
      // then the rotation formula, without repeating the cell’s numeric value.
      out.derivation=[out.lines[0],out.lines[1],...out.lines.slice(3),...out.details,out.lines[2]];
      if(norm)out.notes.push('Head normalization also uses the other components of this head.');
      if(key==='k')out.notes.push('K is rounded when written to the F16 cache.');
    }else{
      add(0,'w'+key,range(r,r,0,cols-1),'b','Weight row');add(index,'x',range(0,cols-1,c,c),'c','Input column');
      if(key!=='v'&&r<(rot??0)&&weights.rope_type!==-1){
        line(eq(mi('u'),row(call('first',mi('b')),mo('·'),mi('c'))));
        line(eq(mi('v'),row(call('second',mi('b')),mo('·'),mi('c'))));
        out.lines.push(...symbolicFrequency(config,'i',rot));
        line(eq(word('angle'),row(word('position'),mo('·'),word('frequency'))));
        line(eq(mi('a'),rotationCases()));
        out.notes.push('This RoPE pairing needs a model-specific explanation.');
      }else line(row(mi('a'),mo(key==='v'||norm?'≈':'='),norm?call('normalize',product('b','c')):product('b','c')));
      if(key==='v')out.notes.push('V is rounded when written to the F16 cache.');
    }
    out.notes.push(`Dot products use all ${cols} components, including the hidden middle.`);
  }else if(key==='scores'){
    out.title='Scaled dot product';
    const keys=links.operands.filter(op=>op.key==='kt'),queries=links.operands.filter(op=>op.key==='q');
    for(const op of keys)add(op.index,op.key,op.range,'b','Key row');
    for(const op of queries)add(op.index,op.key,op.range,'c','Query column');
    line(eq(mi('a'),frac(dot('b','c'),`<msqrt>${mn(trace.model.head_dim)}</msqrt>`)));
    out.notes.push('Both vectors are already post-RoPE. The dot product includes hidden components.');
  }else if(key==='a'){
    out.title='Attention probability';
    for(const op of links.operands)add(op.index,op.key,op.range,'b','Visible scores in this column');
    if(index===0&&r>c){line(eq(mi('a'),call('mask',word('future'))));out.notes.push('Causal masking: a token cannot attend to a future token.');}
    else{
      add(index,'scores',range(r,r,c,c),'c','Selected score');
      line(eq(mi('a'),frac(call('exp',mi('c')),row(mo('∑'),call('exp',row(word('each score in '),mi('b')))))));
      out.notes.push('The denominator includes every causally visible score, including hidden rows.');
    }
  }else if(key==='va'){
    out.title='Weighted value mixture';
    for(const op of links.operands)add(op.index,op.key,op.range,op.key==='a'?'c':'b',op.key==='a'?'Attention column':'Value row');
    line(eq(mi('a'),dot('b','c')));out.notes.push('Multiply matching entries, then add across all visible token positions.');
  }else if(['kt','kc','vc','v_operand'].includes(key)){
    out.title=key==='kt'?'Cached key · transposed':'Cached value';
    for(const op of links.operands)add(op.index,op.key,op.range,'b',op.role);
    line(eq(mi('a'),mi('b')));out.notes.push('The same stored value is reused'+(key==='kt'?' with row and column swapped.':'.'));
  }else{
    out.title=key.startsWith('w')?'Model weight':'Attention input';
    out.notes.push(key.startsWith('w')?'Loaded from the model’s GGUF weights.':'The normalized input entering this attention layer. Its upstream computation is outside this view.');
  }
  return out.lines.length||out.derivation?.length?out:null;
}

// An omitted proxy is a range, never a representative scalar. Reuse the
// exact upstream ranges from the ribbon calculation and keep its math symbolic.
function rangeExplanation(trace,index,key,item,links){
  const out={title:'Selected range',value:null,operands:[],lines:[],notes:[],details:[],
    result:{index,key,range:range(...item.rowRange,...item.colRange)}};
  const add=(op,label)=>out.operands.push({...op,label});
  const multiply=dot('b','c');
  let expression=null,approx=false;
  if(key==='va'){
    for(const op of links.operands)add(op,op.key==='a'?'c':'b');
    expression=multiply;
  }else if(key==='scores'){
    for(const op of links.operands)add(op,op.key==='q'?'c':'b');
    expression=frac(multiply,`<msqrt>${mn(trace.model.head_dim)}</msqrt>`);
  }else if(key==='a'){
    for(const op of links.operands)add(op,'b');
    expression=call('softmax',call('mask',mi('b')));
  }else if(['k','q','v'].includes(key)){
    for(const op of links.operands)add(op,op.key==='x'?'c':'b');
    if(!out.operands.length)return null;
    const transform=trace.weights?.transforms?.[key]||'';
    expression=transform.includes('bias')?row(multiply,mo('+'),word('bias')):multiply;
    if(key!=='v'&&trace.weights?.rope_type!==-1&&item.rowRange[0]<(trace.weights?.rope_dims??Infinity)){
      const config=trace.steps[index].explanation?.[key],dims=config?.dimensions??trace.weights?.rope_dims;
      const paired=ropePair(item.rowRange[0],dims,config?.type??trace.weights?.rope_type);
      const singleRow=item.rowRange[0]===item.rowRange[1];
      const position=item.colRange[0]===item.colRange[1]?mn(item.colRange[0]):mi('j');
      const baseProduct=which=>row(call(which,mi('b')),mo('·'),mi('c'),...(transform.includes('bias')?[mo('+'),word('bias')]:[]));
      out.lines=[math(eq(mi('u'),baseProduct('first'))),math(eq(mi('v'),baseProduct('second'))),
        math(eq(word('position'),row(mn(trace.steps[index].start),mo('+'),position))),
        math(eq(word('pair'),singleRow&&paired?mn(paired.pair):mi('i'))),
        ...symbolicFrequency(config,singleRow&&paired?paired.pair:'i',dims),...rotationGain(config),
        math(eq(word('angle'),row(word('position'),mo('·'),word('frequency'))))];
      let rotated=singleRow&&paired?rotationTerm(paired.secondSelected):rotationCases();
      if(config&&(config.extension!==0||config.magnitude!==1))rotated=row(word('gain'),mo('·'),rotated);
      if(trace.weights?.head_norm)rotated=call('normalize',rotated);
      out.lines.push(math(row(mi('a'),mo(key==='k'?'≈':'='),rotated)));
      return out;
    }
    if(key!=='v'&&trace.weights?.head_norm)expression=call('normalize',expression);
    approx=key==='k'||key==='v';
  }else if(['kt','kc','vc','v_operand'].includes(key)){
    for(const op of links.operands)add(op,'b');
    expression=key==='kt'?call('transpose',mi('b')):mi('b');
  }
  if(!expression)return null;
  out.lines=[math(row(mi('a'),mo(approx?'≈':'='),expression))];
  return out;
}

// All coordinates are screen pixels. The panel stays readable at any scene zoom.
export function placeCellPanel(anchor,panel,bounds){
  if(anchor.right<=bounds.left||anchor.left>=bounds.right||anchor.bottom<=bounds.top||anchor.top>=bounds.bottom)return null;
  const gap=18,clamp=(x,min,max)=>Math.max(min,Math.min(x,Math.max(min,max)));
  const right=anchor.right+gap;
  if(right+panel.width<=bounds.right)
    return {x:right,y:clamp((anchor.top+anchor.bottom-panel.height)/2,bounds.top,bounds.bottom-panel.height)};
  const x=clamp((anchor.left+anchor.right-panel.width)/2,bounds.left,bounds.right-panel.width);
  const below=bounds.bottom-anchor.bottom-gap,above=anchor.top-bounds.top-gap;
  // Prefer below, never the input matrices to the left. Above is only a
  // fallback when the selection is at the viewport's bottom edge.
  if(below>=80||above<80)return {x,y:anchor.bottom+gap};
  return {x,y:Math.max(bounds.top,anchor.top-gap-panel.height)};
}

// Never drop an input label merely because its matrix has left the viewport.
// Offscreen labels become directional navigation targets at the nearest edge.
export function placeOperandIndicator(rect,size,bounds,singleRow=false){
  const clamp=(x,min,max)=>Math.max(min,Math.min(x,Math.max(min,max)));
  const offscreen=rect.bottom<=bounds.top||rect.top>=bounds.bottom||rect.right<=bounds.left||rect.left>=bounds.right;
  let direction='';
  if(offscreen){
    if(rect.bottom<=bounds.top)direction='↑';
    else if(rect.top>=bounds.bottom)direction='↓';
    else if(rect.right<=bounds.left)direction='←';
    else direction='→';
  }
  let x=offscreen?(rect.left+rect.right-size.width)/2:rect.left-size.width-3;
  let y=singleRow?(rect.top+rect.bottom-size.height)/2:rect.top-size.height-3;
  if(direction==='↑')y=bounds.top;
  if(direction==='↓')y=bounds.bottom-size.height;
  if(direction==='←'){x=bounds.left;y=(rect.top+rect.bottom-size.height)/2;}
  if(direction==='→'){x=bounds.right-size.width;y=(rect.top+rect.bottom-size.height)/2;}
  return {x:clamp(x,bounds.left,bounds.right-size.width),y:clamp(y,bounds.top,bounds.bottom-size.height),offscreen,direction};
}

// The mean of the ribbon's two cubic boundaries is its exact centerline.
export function ribbonSample(start,end,vertical,lift,t){
  const delta=(vertical?end.y-start.y:end.x-start.x)*.38;
  const c1=vertical?{x:start.x+lift,y:start.y+delta}:{x:start.x+delta,y:start.y-lift};
  const c2=vertical?{x:end.x+lift,y:end.y-delta}:{x:end.x-delta,y:end.y-lift};
  const u=1-t;
  const x=u*u*u*start.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*end.x;
  const y=u*u*u*start.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*end.y;
  const dx=3*u*u*(c1.x-start.x)+6*u*t*(c2.x-c1.x)+3*t*t*(end.x-c2.x);
  const dy=3*u*u*(c1.y-start.y)+6*u*t*(c2.y-c1.y)+3*t*t*(end.y-c2.y);
  return {x,y,angle:Math.atan2(dy,dx)*180/Math.PI};
}
export function placeRibbonIndicator(start,end,vertical,lift,bounds){
  let best=null,bestScore=Infinity;
  for(let n=12;n<=92;n++){
    const t=n/100,p=ribbonSample(start,end,vertical,lift,t);
    if(p.x<bounds.left+30||p.x>bounds.right-30||p.y<bounds.top+22||p.y>bounds.bottom-36)continue;
    // Prefer a point near the operand, leaving a small gap before its matrix.
    const distance=Math.hypot(p.x-end.x,p.y-end.y),score=Math.abs(distance-28)+(1-t)*5;
    if(score<bestScore){bestScore=score;best={...p,t};}
  }
  return best;
}

// Dock to the operand edge whenever that edge is visible. The ribbon is only
// a fallback when the operand itself has left the viewport.
export function placeOperandEndMarker(start,end,vertical,bounds){
  const point=vertical
    ? {x:end.x,y:end.y+(start.y>end.y?13:-13),angle:start.y>end.y?-90:90}
    : {x:end.x+(start.x>end.x?24:-24),y:end.y,angle:start.x>end.x?180:0};
  if(point.x-20<bounds.left||point.x+20>bounds.right||point.y-12<bounds.top||point.y+12>bounds.bottom)return null;
  return {...point,docked:true};
}

export function isVectorOperand(op){
  return op.vector??(/row|column/i.test(op.role||'')||op.range.r0!==op.range.r1||op.range.c0!==op.range.c1);
}

// Match diagram labels: single bars mark vector operands; scalars stay plain.
export function vectorFormulaMarkup(explanation){
  const vectors=new Set([...(explanation.results||[{...explanation.result,label:'a'}]),...explanation.operands].filter(isVectorOperand).map(op=>op.label));
  return (explanation.derivation||explanation.lines).map(formula=>formula.replace(/<mi mathvariant="normal">([a-e])<\/mi>/g,(markup,label)=>vectors.has(label)?row(mo('|'),markup,mo('|')):markup)).join('');
}

// Matrix-name selections use the actual matrix symbols, without cell variables.
export function matrixFormulaMarkup(trace,key,index=0){
  const name={k:'K',q:'Q',v:'V'},kind={wk:'k',wq:'q',wv:'v'}[key]||key;
  const multiply=(a,b)=>row(mi(a),mo('·'),mi(b));
  let lines=[];
  if(name[kind]){
    let projection=multiply('W'+kind,'X');
    if(trace.weights?.transforms?.[kind]?.includes('bias'))projection=row(projection,mo('+'),word('bias'));
    if(kind!=='v'&&trace.weights?.rope_type!==-1){
      lines.push(math(eq(mi('U'),projection)));
      projection=call('RoPE',row(mi('U'),mo(','),word('position')));
    }
    if(kind!=='v'&&trace.weights?.head_norm)projection=call('normalize',projection);
    lines.push(math(eq(mi(name[kind]),projection)));
  }else if(key==='scores'){
    lines=[math(eq(mi('Scores'),frac(multiply('Kᵀ','Q'),`<msqrt>${mn(trace.model.head_dim)}</msqrt>`)))];
  }else if(key==='a'){
    lines=[math(eq(mi('A'),call('softmax',call('mask',mi('Scores')))))];
  }else if(key==='va'){
    lines=[math(eq(mi('VA'),multiply('V','A')))];
  }else if(key==='kt'){
    lines=[math(eq(mi('Kᵀ'),call('transpose',mi('K cached'))))];
  }else if(['kc','vc','v_operand'].includes(key)){
    const symbol=key==='kc'?'K':'V';
    lines=[math(eq(mi(key==='v_operand'?'V':symbol+' cached'),key==='v_operand'?mi('V cached'):index?call('concat',row(word('previous cache'),mo(','),mi(symbol))):mi(symbol)))];
  }else if(key==='x'){
    lines=[math(eq(mi('X'),call('RMSNorm',word('hidden state'))))];
  }
  return lines.join('');
}

// Two logical pair members can share the same omitted proxy band. Give each
// half of that band its own landing; distinct visible rows retain their edges.
export function pairedRibbonLane(bounds,peer,side){
  if(peer&&['x','y','w','h'].every(k=>Math.abs(bounds[k]-peer[k])<.01))
    return {...bounds,y:bounds.y+side*bounds.h/2,h:bounds.h/2};
  return {...bounds};
}

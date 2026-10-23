import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createRequire} from 'node:module';
import {cellExplanation,ropePair,ropeAngle,placeCellPanel,vectorFormulaMarkup,matrixFormulaMarkup,pairedRibbonLane} from '../frontend/cell-math.mjs';
const require=createRequire(import.meta.url),proxy=require('../web/matrix_proxy.js');
const trace=JSON.parse(gunzipSync(readFileSync(new URL('../demo/data/layer-0-head-0.json.gz',import.meta.url))));
function explanation(key,r,c,index=0,t=trace){
  const timeline=proxy.layout(t,{compact:true,showProjections:true}),block=timeline.columns[index].blocks[key];
  const p=block.plan,item=proxy.cellAt(p,p.rowIndices.indexOf(r),p.colIndices.indexOf(c));
  return cellExplanation(t,timeline,index,key,item,proxy.mathLinks(t,timeline,index,key,item));
}
test('RoPE input labels match first/second paired rows and preserve the selected input column',()=>{
  for(const r of [0,1,62,63]){
    const out=explanation('k',r,1),pair=ropePair(r,64,0);
    assert.equal(out.operands[0].label,'c');assert.equal(out.operands[0].range.r0,pair.first);
    assert.equal(out.operands[1].label,'d');assert.equal(out.operands[1].range.r0,pair.second);
    assert.equal(out.operands[2].label,'e');assert.equal(out.operands[2].range.c0,1);
    assert.ok(out.lines.every(s=>s.startsWith('<math')));
    assert.ok(out.details.join('').includes('100000'));
    assert.match(out.notes.join(' '),/F16/);
  }
  assert.deepEqual(ropePair(33,64,2),{first:1,second:33,pair:1,secondSelected:true});
  assert.equal(ropePair(64,64,0),null);assert.equal(ropePair(0,64,8),null);
});
test('Captured RoPE intermediates reproduce every first-head K component within F16 rounding',()=>{
  for(const step of trace.steps){
    for(let r=0;r<64;r++)for(let c=0;c<step.count;c++){
      const cfg=step.explanation.k,pair=ropePair(r,cfg.dimensions,cfg.type),theta=ropeAngle(cfg,pair.pair,step.start+c).angle;
      const u=cfg.before[pair.first][c],v=cfg.before[pair.second][c];
      const expected=pair.secondSelected?u*Math.sin(theta)+v*Math.cos(theta):u*Math.cos(theta)-v*Math.sin(theta);
      assert.ok(Math.abs(expected-step.matrices.k[r][c])<=Math.max(.00001,Math.abs(expected)*.0005),`${r}, ${c}: ${expected}`);
    }
  }
});
test('Angle uses model base, pair, position, scale and frequency factors; never invents extended scaling',()=>{
  const cfg={type:2,dimensions:64,base:500000,scale:.5,extension:0,magnitude:1,factors:[1,2]};
  assert.equal(ropeAngle(cfg,0,7).angle,3.5);
  assert.equal(ropeAngle(cfg,1,7).angle,7*Math.pow(500000,-2/64)*.5/2);
  assert.equal(ropeAngle({...cfg,extension:1},0,7),null);assert.equal(ropeAngle(undefined,0,7),null);
  const legacy=structuredClone(trace);delete legacy.steps[0].explanation;
  const out=explanation('k',0,1,0,legacy);
  assert.ok(out.notes.some(s=>s.includes('does not include RoPE')));
  assert.ok(!out.details.join('').includes('100000'));
});
test('Scores, softmax, weighted sums and cache copies use the same result/input labels',()=>{
  const scores=explanation('scores',2,1);
  assert.deepEqual(scores.operands.map(o=>[o.key,o.label]),[['kt','b'],['q','c']]);
  assert.ok(scores.lines.join('').includes('<mfrac>'));assert.ok(scores.lines.join('').includes('<msqrt>'));
  const softmax=explanation('a',1,2);assert.deepEqual(softmax.operands.map(o=>o.label),['b','c']);
  assert.match(explanation('a',3,1).notes.join(' '),/Causal masking/);
  const va=explanation('va',0,1);assert.deepEqual(va.operands.map(o=>o.label),['b','c']);
  assert.match(explanation('kt',0,0).title,/transposed/);
});
test('Formula panel chooses the available side, remains bounded, and hides when its cell leaves view',()=>{
  const bounds={left:10,right:800,top:20,bottom:600},panel={width:300,height:200};
  assert.deepEqual(placeCellPanel({left:100,right:130,top:200,bottom:230},panel,bounds),{x:148,y:115});
  assert.deepEqual(placeCellPanel({left:700,right:730,top:200,bottom:230},panel,bounds),{x:500,y:248});
  const edge=placeCellPanel({left:0,right:20,top:0,bottom:30},panel,bounds);assert.equal(edge.y,20);
  assert.equal(placeCellPanel({left:900,right:930,top:200,bottom:230},panel,bounds),null);
});

test('Input labels stay at viewport edges with the correct direction when operands are offscreen',async()=>{
  const {placeOperandIndicator}=await import('../frontend/cell-math.mjs');
  const bounds={left:10,right:800,top:100,bottom:600},size={width:150,height:26};
  const above=placeOperandIndicator({left:400,right:450,top:-400,bottom:50},size,bounds);
  assert.equal(above.offscreen,true);assert.equal(above.direction,'↑');assert.equal(above.y,100);
  const below=placeOperandIndicator({left:400,right:450,top:700,bottom:800},size,bounds);
  assert.equal(below.direction,'↓');assert.equal(below.y,574);
  assert.equal(placeOperandIndicator({left:-400,right:-10,top:200,bottom:240},size,bounds,true).direction,'←');
  assert.equal(placeOperandIndicator({left:900,right:1200,top:200,bottom:240},size,bounds,true).direction,'→');
  const partial=placeOperandIndicator({left:400,right:450,top:-400,bottom:300},size,bounds);
  assert.equal(partial.offscreen,false);assert.equal(partial.y,100);
  for(const rect of [{left:-1000,right:-900,top:-1000,bottom:-900},{left:900,right:1000,top:900,bottom:1000}]){
    const p=placeOperandIndicator(rect,size,bounds);assert.ok(p.x>=10&&p.x+150<=800);assert.ok(p.y>=100&&p.y+26<=600);
  }
});

test('Ribbon arrows follow the actual tape centerline toward inputs, including offscreen inputs',async()=>{
  const {ribbonSample,placeRibbonIndicator}=await import('../frontend/cell-math.mjs');
  const start={x:600,y:500},end={x:100,y:500},bounds={left:0,right:900,top:100,bottom:700};
  const midpoint=ribbonSample(start,end,false,50,.5);
  assert.equal(midpoint.x,350);assert.equal(midpoint.y,462.5);
  const left=placeRibbonIndicator(start,end,false,50,bounds);
  assert.ok(left.x>100&&left.x<200);assert.ok(Math.abs(left.angle)>150);
  const up=placeRibbonIndicator(start,{x:650,y:-400},true,50,bounds);
  assert.ok(up);assert.ok(up.y>=122&&up.y<=200);assert.ok(up.angle<0);
  const p=ribbonSample(start,{x:650,y:-400},true,50,up.t);
  assert.deepEqual({x:up.x,y:up.y},{x:p.x,y:p.y});
  assert.equal(placeRibbonIndicator({x:-600,y:-500},{x:-100,y:-200},false,40,bounds),null);
});

test('Visible column markers dock immediately below the column rather than midway along its ribbon',async()=>{
  const {placeOperandEndMarker}=await import('../frontend/cell-math.mjs');
  const bounds={left:0,right:1000,top:100,bottom:800};
  assert.deepEqual(placeOperandEndMarker({x:550,y:700},{x:500,y:400},true,bounds),{x:500,y:413,angle:-90,docked:true});
  assert.deepEqual(placeOperandEndMarker({x:700,y:500},{x:300,y:500},false,bounds),{x:324,y:500,angle:180,docked:true});
  assert.equal(placeOperandEndMarker({x:550,y:700},{x:500,y:-400},true,bounds),null);
});

test('Notebook derivation shows position and frequency before rotation without a final numeric result',()=>{
  const out=explanation('k',0,1),steps=out.derivation;
  assert.ok(steps&&steps.length>out.lines.length);
  const position=steps.findIndex(s=>s.includes('<mtext>position</mtext>'));
  const frequency=steps.findIndex(s=>s.includes('<msup>'));
  const angle=steps.findIndex(s=>s.includes('<mtext>angle</mtext><mo>=</mo>'));
  const rotation=steps.length-1;
  assert.ok(position>1&&position<frequency&&frequency<angle&&angle<rotation);
  assert.ok(steps[position].includes('<mn>0</mn><mo>+</mo><mn>1</mn>'));
  assert.ok(steps[rotation].includes('cos'));
  assert.match(steps.at(-1),/cos/);
  assert.ok(!steps.some(s=>/<mi mathvariant="normal">a<\/mi><mo>=<\/mo><mn>/.test(s)));
  const legacy=structuredClone(trace);delete legacy.steps[0].explanation;
  const old=explanation('k',0,1,0,legacy);
  assert.match(old.derivation.at(-1),/cos/);
  assert.ok(old.derivation.some(s=>s.includes('<mtext>position</mtext>')));
});

test('Omitted VA ranges get a symbolic matrix-product card with the correct result and operand ranges',()=>{
  const timeline=proxy.layout(trace,{compact:true,showProjections:true}),p=timeline.columns[0].blocks.va.plan;
  const item=proxy.cellAt(p,p.rowIndices.indexOf(null),3),links=proxy.mathLinks(trace,timeline,0,'va',item);
  assert.ok(item.omitted);
  const out=cellExplanation(trace,timeline,0,'va',item,links);
  assert.deepEqual(out.result.range,{r0:item.rowRange[0],r1:item.rowRange[1],c0:3,c1:3});
  assert.equal(out.value,null);assert.deepEqual(out.operands.map(op=>op.label),['b','c']);
  assert.match(out.lines[0],/<mo[^>]*>·<\/mo>/);assert.ok(!out.lines[0].includes('<mn>'));
});

test('Omitted projection, attention and cache ranges show formulas without invented cell values',()=>{
  const timeline=proxy.layout(trace,{compact:true,showProjections:true});
  for(const key of ['k','q','v','kt','v_operand']){
    const p=timeline.columns[0].blocks[key].plan;
    const ri=p.rowIndices.includes(null)?p.rowIndices.indexOf(null):0;
    const ci=p.colIndices.includes(null)?p.colIndices.indexOf(null):0;
    const item=proxy.cellAt(p,ri,ci);assert.ok(item.omitted,key);
    const out=cellExplanation(trace,timeline,0,key,item,proxy.mathLinks(trace,timeline,0,key,item));
    assert.ok(out?.lines.length,key);assert.equal(out.value,null,key);
    assert.ok(!out.lines.some(line=>/<mi mathvariant="normal">a<\/mi><mo>=<\/mo><mn>/.test(line)),key);
  }
});

test('Missing RoPE metadata still exposes the frequency and rotation calculation with named intermediates',()=>{
  const old=structuredClone(trace);delete old.steps[0].explanation;
  const out=explanation('k',2,3,0,old),derivation=out.derivation.join('');
  assert.match(derivation,/<msup>/);assert.match(derivation,/<mtext>base<\/mtext>/);
  assert.match(derivation,/<mtext>frequency<\/mtext>/);assert.match(derivation,/cos/);assert.match(derivation,/sin/);
  assert.ok(!derivation.includes('>RoPE<'));
});

test('Omitted RoPE ranges expand the two paired dot products and rotations instead of wrapping a product in RoPE',()=>{
  const timeline=proxy.layout(trace,{compact:true,showProjections:true}),p=timeline.columns[0].blocks.k.plan;
  const item=proxy.cellAt(p,p.rowIndices.indexOf(null),3),links=proxy.mathLinks(trace,timeline,0,'k',item);
  const out=cellExplanation(trace,timeline,0,'k',item,links),formulas=out.derivation.join('');
  assert.ok(!formulas.includes('>first<'));assert.ok(!formulas.includes('>second<'));assert.match(formulas,/<msup>/);
  assert.match(formulas,/<mi mathvariant="normal">i<\/mi>/);assert.match(formulas,/cos/);assert.match(formulas,/sin/);
  assert.ok(!formulas.includes('>RoPE<'));assert.equal(out.value,null);
});

test('RoPE derivations keep intermediates symbolic and only substitute known parameters',()=>{
  const out=explanation('k',2,1),steps=out.derivation,joined=steps.join('');
  assert.ok(!steps.some(s=>/<mi mathvariant="normal">[uv]<\/mi><mo>=<\/mo><mn>/.test(s)));
  assert.ok(!steps.some(s=>/<mtext>(frequency|angle)<\/mtext><mo>=<\/mo><mn>/.test(s)));
  assert.ok(!steps.some(s=>s.includes('>cos<')&&s.includes('<mo>=</mo><mn>')));
  assert.ok(!steps.some(s=>s.includes('>sin<')&&s.includes('<mo>=</mo><mn>')));
  assert.match(joined,/100000/);assert.match(joined,/<msup>/);
  assert.ok(joined.includes('<mtext>position</mtext><mo class="math-product" lspace="0.25em" rspace="0.25em">·</mo><mtext>frequency</mtext>'));
});

test('formula bars distinguish the softmax vector from its scalar score',()=>{
  const markup=vectorFormulaMarkup(explanation('a',1,2));
  assert.ok(markup.includes('<mo>|</mo><mi mathvariant="normal">b</mi><mo>|</mo>'));
  assert.ok(!markup.includes('<mo>|</mo><mi mathvariant="normal">c</mi>'));
  assert.ok(!markup.includes('<mo>|</mo><mi mathvariant="normal">a</mi>'));
});
test('both paired weight rows and the input column use bars in RoPE formulas',()=>{
  const markup=vectorFormulaMarkup(explanation('k',1,2));
  for(const label of ['c','d','e'])assert.ok(markup.includes(`<mo>|</mo><mi mathvariant="normal">${label}</mi><mo>|</mo>`));
  assert.ok(!markup.includes('<mo>|</mo><mi mathvariant="normal">u</mi>'));
});

test('matrix formulas use matrix names and preserve the post-RoPE projection stage',()=>{
  const v=matrixFormulaMarkup(trace,'v');
  assert.match(v,/>V</);assert.match(v,/>Wv</);assert.match(v,/>X</);assert.ok(!v.includes('RoPE'));
  assert.match(matrixFormulaMarkup(trace,'k'),/RoPE/);
  assert.equal(matrixFormulaMarkup(trace,'wv'),v);
  assert.match(matrixFormulaMarkup(trace,'scores'),/msqrt/);
  assert.match(matrixFormulaMarkup(trace,'a'),/softmax/);
  assert.match(matrixFormulaMarkup(trace,'va'),/>VA</);
  assert.match(matrixFormulaMarkup(trace,'kc',1),/concat/);
});

test('RoPE uses the same complete two-output rotation from either member of a pair',()=>{
  const first=explanation('k',0,1),second=explanation('k',1,1);
  assert.equal(first.derivation.at(-1),second.derivation.at(-1));
  assert.deepEqual(first.results.map(op=>[op.label,op.range.r0]),[['a',0],['b',1]]);
  const formula=first.derivation.at(-1);
  assert.equal((formula.match(/<mtable /g)||[]).length,3);
  assert.ok(formula.includes('<mi mathvariant="normal">a</mi>'));assert.ok(formula.includes('<mi mathvariant="normal">b</mi>'));
  assert.ok(formula.includes('<mo>−</mo>'));
});

test('RoPE frequency uses a true superscript rather than power function notation',()=>{
  for(const t of [trace,(()=>{const copy=structuredClone(trace);delete copy.steps[0].explanation;return copy;})()]){
    const markup=explanation('k',2,1,0,t).derivation.join('');
    assert.match(markup,/<msup>/);assert.match(markup,/<mfrac>/);assert.ok(!markup.includes('>power<'));
  }
});

test('omitted RoPE pairs use separate variables and exact paired row sets',()=>{
  const timeline=proxy.layout(trace,{compact:true,showProjections:true}),p=timeline.columns[0].blocks.k.plan;
  const item=proxy.cellAt(p,p.rowIndices.indexOf(null),2);
  const out=cellExplanation(trace,timeline,0,'k',item,proxy.mathLinks(trace,timeline,0,'k',item));
  assert.deepEqual(out.results.map(op=>op.label),['a','b']);
  assert.deepEqual(out.operands.map(op=>op.label),['c','d','e']);
  assert.ok(out.results[0].ranges.every(r=>r.r0%2===0));
  assert.ok(out.results[1].ranges.every(r=>r.r0%2===1));
  assert.ok(!/>(first|second)</.test(out.derivation.join('')));
});
test('split-half RoPE selects the actual separated partner rather than the adjacent row',()=>{
  const t=structuredClone(trace);t.steps[0].explanation.k.type=2;
  const timeline=proxy.layout(t,{compact:true,showProjections:true}),item={row:33,col:1,rowRange:[33,33],colRange:[1,1],omitted:false};
  const out=cellExplanation(t,timeline,0,'k',item,proxy.mathLinks(t,timeline,0,'k',item));
  assert.deepEqual(out.results.map(op=>op.range.r0),[1,33]);
  assert.deepEqual(out.operands.slice(0,2).map(op=>op.range.r0),[1,33]);
});

test('paired ribbon lanes preserve separate visible rows and split shared omissions',()=>{
  const top={x:100,y:200,w:38,h:24},bottom={...top,y:224};
  assert.deepEqual(pairedRibbonLane(top,bottom,0),top);
  assert.deepEqual(pairedRibbonLane(bottom,top,1),bottom);
  const omitted={x:100,y:200,w:38,h:48};
  const a=pairedRibbonLane(omitted,omitted,0),b=pairedRibbonLane(omitted,omitted,1);
  assert.equal(a.y+a.h,b.y);assert.equal(a.h,24);assert.equal(b.h,24);
  assert.equal(b.y+b.h,omitted.y+omitted.h);
});

test('formula panel falls below a pair when the right side is full, with above as bottom-edge fallback',()=>{
  const bounds={left:10,right:800,top:20,bottom:600},panel={width:330,height:280};
  const pair={left:700,right:740,top:200,bottom:250};
  const below=placeCellPanel(pair,panel,bounds);
  assert.equal(below.y,268);assert.ok(below.y>pair.bottom);
  const bottom={...pair,top:540,bottom:590};
  const above=placeCellPanel(bottom,panel,bounds);
  assert.ok(above.y+panel.height<bottom.top);
});

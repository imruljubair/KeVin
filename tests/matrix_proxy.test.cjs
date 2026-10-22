const {test}=require('node:test');
const assert=require('node:assert/strict');
const {plan,extendPlan,cellAt,hitTest,formatNumber,layout,regions,mathLinks,weightValues,routeConnection,curvedConnection,ribbonConnection,mergeRegions,connectionAnchors,rangeOutline}=require('../web/matrix_proxy.js');
const {matrixRange}=require('../web/matrix_proxy.js');

test('approved 32 x 64 proxy retains exactly the original four corner slices',()=>{
  const p=plan(32,64);
  assert.deepEqual(p.rowIndices,[0,1,2,null,29,30,31]);
  assert.deepEqual(p.colIndices,[0,1,2,3,null,60,61,62,63]);
  assert.deepEqual(p.rowGap,[3,28]);assert.deepEqual(p.colGap,[4,59]);
  assert.equal(p.retained,48);
});
test('transposition swaps selected original indices and preserves shape direction',()=>{
  const wide=plan(32,64),tall=plan(64,32);
  assert.deepEqual(wide.rowIndices,tall.colIndices);
  assert.deepEqual(wide.colIndices,tall.rowIndices);
  assert.ok(wide.width>tall.width);assert.ok(tall.height>wide.height);
});
test('corner sampling caps each edge at four cells while keeping small axes intact',()=>{
  for(const [rows,cols] of [[64,576],[576,64],[64,8],[8,64],[64,1],[1,64]]){
    const p=plan(rows,cols);
    assert.ok(p.edgeRows<=4&&p.edgeCols<=4);
    for(const indices of [p.rowIndices,p.colIndices])if(indices.includes(null)){
      assert.ok(indices.indexOf(null)<=4);
      assert.ok(indices.length-indices.indexOf(null)-1<=4);
    }
  }
  assert.deepEqual(plan(64,4).colIndices,[0,1,2,3]);
});
test('small matrices retain all cells; vectors have no invented rows or columns',()=>{
  const p=plan(4,4);
  assert.deepEqual(p.rowIndices,[0,1,2,3]);assert.deepEqual(p.colIndices,[0,1,2,3]);
  assert.equal(p.rowGap,null);assert.equal(p.colGap,null);assert.equal(p.retained,16);
  for(const [rows,cols] of [[1,64],[64,1],[1,1]]){
    const v=plan(rows,cols);
    assert.equal(v.rowIndices.filter(x=>x!==null).length,new Set(v.rowIndices.filter(x=>x!==null)).size);
    assert.equal(v.colIndices.filter(x=>x!==null).length,new Set(v.colIndices.filter(x=>x!==null)).size);
    assert.ok(v.rowIndices.every(x=>x===null||x>=0&&x<rows));
    assert.ok(v.colIndices.every(x=>x===null||x>=0&&x<cols));
  }
});
test('hit-testing returns original indices, never an average or a gap value',()=>{
  const p=plan(32,64),matrix=Array.from({length:32},(_,r)=>Array.from({length:64},(_,c)=>r*64+c+.125));
  for(let ri=0;ri<p.rowIndices.length;ri++)for(let ci=0;ci<p.colIndices.length;ci++){
    const item=hitTest(p,p.padX+(ci+.5)*p.cellWidth,p.padY+(ri+.5)*p.cellHeight);
    if(item.omitted){assert.ok(item.row===null||item.col===null||item.diamondHidden);assert.equal('value' in item,false);}
    else assert.equal(matrix[item.row][item.col],item.row*64+item.col+.125);
  }
  const last=hitTest(p,p.width-2,p.height-2);assert.equal(last.row,31);assert.equal(last.col,63);
  assert.equal(hitTest(p,0,0),null);
});
test('fading approaches the omitted center and is symmetric across corners',()=>{
  const p=plan(32,64);
  assert.equal(cellAt(p,0,0).fade,1);
  assert.ok(cellAt(p,1,2).fade<=cellAt(p,0,0).fade);
  assert.ok(cellAt(p,2,3).fade<cellAt(p,1,2).fade);
  assert.equal(cellAt(p,2,3).fade,cellAt(p,4,5).fade);
  assert.equal(cellAt(plan(4,4),2,2).fade,1);
});
test('dimensions grow in bounded space rather than allocating the hidden center',()=>{
  const small=plan(64,9),large=plan(576,9),huge=plan(100000,100000);
  assert.equal(large.cellHeight,small.cellHeight);assert.equal(large.cellWidth,small.cellWidth);
  for(const p of [small,large,huge]){assert.ok(p.rowIndices.length<=13);assert.ok(p.colIndices.length<=13);assert.ok(p.width<1000);assert.ok(p.height<650);}
});
test('worksheet values use the littleLLAMA two-decimal display without altering source data',()=>{
  assert.equal(formatNumber(0),'0.00');assert.equal(formatNumber(-.23),'-0.23');
  assert.equal(formatNumber(.00012),'0.00');assert.equal(formatNumber(-.00012),'0.00');
  assert.equal(formatNumber(-.02),'-0.02');assert.equal(formatNumber(1.2),'1.20');
  assert.equal(formatNumber(123.456),'123.46');assert.equal(formatNumber(NaN),'—');
});

function fixture(prefill=9,decode=2){
  const m=(rows,cols)=>Array.from({length:rows},(_,r)=>Array.from({length:cols},(_,c)=>r*cols+c));
  return {steps:[prefill,...Array(decode).fill(1)].map((count,index)=>{
    const start=index?prefill-1+index:0,n=start+count;
    return {start,count,matrices:{x:m(576,count),k:m(64,count),v:m(64,count),q:m(64,count),kt:m(n,64),a:m(n,count),va:m(64,count),kc:m(64,n),vc:m(64,n)}};
  })};
}
test('reference layout aligns K transpose with Scores, V with VA, with no duplicate cache section',()=>{
  const t=layout(fixture());
  for(const col of t.columns){
    const b=col.blocks;
    assert.equal(b.kt.y,b.scores.y+(col.index?b.scores.plan.rowIndices.indexOf(col.step.start)*b.scores.plan.cellHeight:0));assert.equal(b.v_operand.y,b.va.y);
    assert.ok(b.kt.x+b.kt.w<b.a.x);assert.ok(b.v_operand.x+b.v_operand.w<b.va.x);
    assert.equal(b.x.x,b.k.x);assert.equal(b.k.x,b.q.x);assert.equal(b.q.x,b.a.x);assert.equal(b.q.x,b.scores.x);assert.ok(b.a.y>b.scores.y+b.scores.h);
    assert.ok(b.k.y>b.x.y+b.x.h);assert.ok(b.v.y>b.k.y+b.k.h);assert.ok(b.q.y>b.v.y+b.v.h);
    assert.equal(b.kc,undefined);assert.equal(b.vc,undefined);assert.deepEqual(t.groups.map(g=>g.title),['Input','Projections','Attention','Value aggregation']);
    for(const block of Object.values(b)){assert.ok(block.x>=col.x);assert.ok(block.x+block.w<col.x+col.width);assert.ok(block.y+block.h<t.height);}
  }
});
test('compact mode shares actual latest caches without changing earlier attention data',()=>{
  const trace=fixture(),t=layout(trace,{compact:true});
  assert.equal(t.columns[0].blocks.kt.values,trace.steps[2].matrices.kt);
  assert.equal(t.columns[0].blocks.v_operand.values,trace.steps[2].matrices.vc);
  assert.equal(t.columns[0].blocks.v_operand.sourceStep,2);
  assert.equal(t.columns[0].blocks.v_operand.appendStart,9);
  for(const col of t.columns){
    assert.equal(col.blocks.a.values,trace.steps[col.index].matrices.a);
    assert.equal(col.blocks.va.values,trace.steps[col.index].matrices.va);
    if(col.index){assert.equal(col.blocks.kt,undefined);assert.equal(col.blocks.v_operand,undefined);assert.equal(col.blocks.kc,undefined);}
  }
});
test('appended cache positions remain in the retained end slices',()=>{
  for(const n of [9,16,32,64,127,128]){
    const p=plan(64,n),last=cellAt(p,p.rowIndices.length-1,p.colIndices.length-1);
    assert.equal(last.row,63);assert.equal(last.col,n-1);assert.equal(last.omitted,false);
  }
});
test('collapsing projections preserves all attention and cache operands',()=>{
  const trace=fixture(),full=layout(trace),collapsed=layout(trace,{showProjections:false});
  assert.ok(collapsed.height<full.height);
  for(const col of collapsed.columns){
    assert.equal(col.blocks.x.values,full.columns[col.index].blocks.x.values);assert.equal(col.blocks.q.values,full.columns[col.index].blocks.q.values);
    for(const key of ['kt','a','va','v_operand'])assert.equal(col.blocks[key].values,full.columns[col.index].blocks[key].values);
  }
});

test('prefill samples, omission bands, spacing and fading stay fixed as every generated token is appended',()=>{
  for(const [rows,cols,axis] of [[32,64,'row'],[64,32,'col'],[4,64,'row']]){
    const base=plan(rows,cols),grown=extendPlan(base,axis==='row'?128:rows,axis==='col'?128:cols);
    assert.deepEqual(grown.rowGap,base.rowGap);assert.deepEqual(grown.colGap,base.colGap);
    const initial=axis==='row'?rows:cols,indices=axis==='row'?grown.rowIndices:grown.colIndices;
    assert.deepEqual(indices.slice(0,axis==='row'?base.rowIndices.length:base.colIndices.length),axis==='row'?base.rowIndices:base.colIndices);
    for(let n=initial;n<128;n++)assert.ok(indices.includes(n));
    assert.equal(grown.cellHeight,base.cellHeight);assert.equal(grown.cellWidth,base.cellWidth);
    for(let r=0;r<base.rowIndices.length;r++)for(let c=0;c<base.colIndices.length;c++)assert.equal(cellAt(grown,r,c).fade,cellAt(base,r,c).fade);
  }
});
test('non-compact decode draws only appended K/V components, with original coordinates and values',()=>{
  const trace=fixture(32,12),t=layout(trace,{showProjections:false});
  for(const col of t.columns.slice(1)){
    for(const key of ['kt','v_operand']){
      const b=col.blocks[key],indices=key==='kt'?b.plan.rowIndices:b.plan.colIndices;
      assert.deepEqual(indices,[col.step.start]);assert.equal(b.appendOnly,true);
      assert.equal(b.values,trace.steps[col.index].matrices[key==='v_operand'?'vc':key]);
      const cell=cellAt(b.plan,0,0);assert.equal(key==='kt'?cell.row:cell.col,col.step.start);
    }
    const a=col.blocks.a.plan;
    for(let i=32;i<=col.step.start;i++)assert.ok(a.rowIndices.includes(i));
    assert.deepEqual(a.rowGap,t.columns[0].blocks.a.plan.rowGap);
  }
});
test('compact mode preserves the prefill proxy then shows every appended row and column',()=>{
  const trace=fixture(32,20),t=layout(trace,{compact:true});
  for(const key of ['kt','v_operand']){
    const b=t.columns[0].blocks[key],initial=layout({steps:[trace.steps[0]]}).columns[0].blocks[key].plan;
    const before=key==='kt'?initial.rowIndices:initial.colIndices,after=key==='kt'?b.plan.rowIndices:b.plan.colIndices;
    assert.deepEqual(after,[...before,...Array.from({length:20},(_,i)=>32+i)]);
    assert.equal(b.appendOnly,false);
  }
});

test('compact appends keep K fixed and grow V leftward with a stable right edge',()=>{
  const trace=fixture(4,12),before=layout({steps:[trace.steps[0]]},{compact:true}),after=layout(trace,{compact:true});
  const first=before.columns[0].blocks,last=after.columns[0].blocks;
  assert.equal(first.kt.x,last.kt.x);
  assert.equal(first.v_operand.x+first.v_operand.w,last.v_operand.x+last.v_operand.w);
  assert.ok(last.v_operand.x<first.v_operand.x);
  assert.ok(after.height<after.divider+30);
});

test('math ranges explicitly distinguish retained operands from omitted spans',()=>{
  const p=plan(32,64),r=regions(p,{r0:1,r1:30,c0:2,c1:61});
  assert.ok(r.some(x=>x.hidden));assert.ok(r.some(x=>!x.hidden));
  assert.equal(regions(p,{r0:32,r1:32,c0:0,c1:63}).length,0);
  assert.equal(regions(p,{r0:0,r1:0,c0:0,c1:0})[0].hidden,false);
});
test('decode softmax links to its Scores column; raw Scores link only the selected key row and query',()=>{
  const trace=fixture(32,5),t=layout(trace,{showProjections:false});
  const math=mathLinks(trace,t,3,'a',{rowRange:[2,2],colRange:[0,0]});
  assert.match(math.formula,/softmax/);
  assert.deepEqual(math.operands.map(op=>op.key),['scores']);
  assert.deepEqual(math.operands[0].range,{r0:0,r1:34,c0:0,c1:0});
  const raw=mathLinks(trace,t,3,'scores',{rowRange:[2,2],colRange:[0,0]});
  assert.ok(raw.operands.some(op=>op.key==='q'));
  assert.deepEqual(raw.operands.filter(op=>op.key==='kt').map(op=>op.range.r0),[2]);
  assert.ok(raw.operands.every(op=>op.index<=3));
});
test('VA sum uses the correct value row and attention column, including hidden components',()=>{
  const trace=fixture(32,5),t=layout(trace,{compact:true});
  const math=mathLinks(trace,t,2,'va',{rowRange:[63,63],colRange:[0,0]});
  const v=math.operands.find(x=>x.key==='v_operand'),a=math.operands.find(x=>x.key==='a');
  assert.equal(v.index,0);assert.deepEqual(v.range,{r0:63,r1:63,c0:0,c1:33});
  assert.equal(a.index,2);assert.deepEqual(a.range,{r0:0,r1:33,c0:0,c1:0});
  assert.ok(regions(t.columns[0].blocks.v_operand.plan,v.range).some(x=>x.hidden));
});
test('cache transpose math preserves original appended token coordinates',()=>{
  const trace=fixture(32,4),t=layout(trace);
  const math=mathLinks(trace,t,4,'kt',{rowRange:[35,35],colRange:[63,63]});
  assert.equal(math.operands.length,1);assert.equal(math.operands[0].index,4);assert.equal(math.operands[0].key,'k');
  assert.deepEqual(math.operands[0].range,{r0:63,r1:63,c0:0,c1:0});
});
test('non-compact cache names connect only their displayed token strip',()=>{
  const trace=fixture(8,3),t=layout(trace,{compact:false});
  for(const col of t.columns)for(const key of ['kt','v_operand']){
    const b=col.blocks[key],range=matrixRange(b.plan),links=mathLinks(trace,t,col.index,key,range);
    assert.deepEqual(links.operands.map(op=>op.index),[col.index]);
    assert.deepEqual(links.operands.map(op=>op.key),[key==='kt'?'k':'v']);
    if(col.index)assert.deepEqual(key==='kt'?range.rowRange:range.colRange,[col.step.start,col.step.start+col.step.count-1]);
  }
  const compact=layout(trace,{compact:true});
  for(const key of ['kt','v_operand']){
    const b=compact.columns[0].blocks[key],links=mathLinks(trace,compact,0,key,matrixRange(b.plan));
    assert.deepEqual(links.operands.map(op=>op.index),trace.steps.map((_,i)=>i));
  }
});
test('causally masked attention has no operand connectors',()=>{
  const trace=fixture(32,1),t=layout(trace);
  const math=mathLinks(trace,t,0,'a',{rowRange:[31,31],colRange:[0,0]});
  assert.match(math.formula,/causal mask/);assert.deepEqual(math.operands,[]);
});

function withWeights(trace=fixture(32,3)){
  const p=plan(64,576),ri=[0,1,2,3,4,5,58,59,60,61,62,63],ci=p.colIndices.filter(x=>x!==null);
  const payload=()=>({rows:64,cols:576,row_indices:ri,col_indices:ci,values:ri.map(r=>ci.map(c=>r+c/1000)),storage_type:'q8_0',head:0});
  trace.weights={wq:payload(),wk:payload(),wv:payload(),rope_dims:64,rope_type:0,head_norm:false,transforms:{q:'linear → RoPE',k:'linear → RoPE → F16 cache',v:'linear → F16 cache'}};
  return trace;
}
test('inner sections contain each matrix once, with caches in attention and output',()=>{
  for(const compact of [false,true]){
    const t=layout(withWeights(fixture(3,2)),{compact});
    const expected={x:'Input',wk:'Projections',wv:'Projections',wq:'Projections',k:'Projections',v:'Projections',q:'Projections',kt:'Attention',scores:'Attention',a:'Attention',v_operand:'Value aggregation',va:'Value aggregation'};
    for(const col of t.columns)for(const b of Object.values(col.blocks)){
      const groups=t.groups.filter(g=>b.y>=g.top&&b.y+b.h<=g.top+g.height);
      assert.deepEqual(groups.map(g=>g.title),[expected[b.key]]);
    }
  }
});
test('weight payload populates only real retained components and preserves original dimensions',()=>{
  const trace=withWeights(),values=weightValues(trace.weights.wq),p=plan(64,576);
  assert.equal(values.length,64);assert.equal(values[0].length,576);
  for(const r of p.rowIndices)for(const c of p.colIndices)if(r!==null&&c!==null)assert.equal(values[r][c],r+c/1000);
  assert.equal(values[6][0],undefined);assert.equal(values[0][6],undefined);
});
test('projection weights appear once beside their matching outputs; X remains visible when collapsed',()=>{
  const trace=withWeights(),full=layout(trace),small=layout(trace,{showProjections:false});
  for(const kind of ['q','k','v']){
    const w=full.columns[0].blocks['w'+kind],output=full.columns[0].blocks[kind];
    assert.equal(w.y,output.y);assert.ok(w.x+w.w<output.x);assert.equal(w.parameter,true);
    for(const c of full.columns.slice(1))assert.equal(c.blocks['w'+kind],undefined);
  }
  for(const col of small.columns){assert.ok(col.blocks.x);assert.ok(col.blocks.q);assert.equal(col.blocks.wq,undefined);}
});
test('projection math connects the full X column and correct RoPE paired weight rows',()=>{
  const trace=withWeights(),t=layout(trace),math=mathLinks(trace,t,2,'q',{rowRange:[0,0],colRange:[0,0]});
  const w=math.operands.find(op=>op.key==='wq'),x=math.operands.find(op=>op.key==='x');
  assert.equal(w.index,0);assert.deepEqual(w.range,{r0:0,r1:1,c0:0,c1:575});
  assert.equal(x.index,2);assert.deepEqual(x.range,{r0:0,r1:575,c0:0,c1:0});
  assert.ok(regions(t.columns[0].blocks.wq.plan,w.range).some(r=>r.hidden));
  trace.weights.rope_type=2;
  const neox=mathLinks(trace,t,2,'q',{rowRange:[0,0],colRange:[0,0]});
  assert.deepEqual(neox.operands.filter(op=>op.key==='wq').map(op=>[op.range.r0,op.range.r1]),[[0,0],[32,32]]);
  trace.weights.head_norm=true;
  assert.deepEqual(mathLinks(trace,t,2,'q',{rowRange:[0,0],colRange:[0,0]}).operands.find(op=>op.key==='wq').range,{r0:0,r1:63,c0:0,c1:575});
});
test('V projection math uses one weight row without adding a RoPE partner',()=>{
  const trace=withWeights(),t=layout(trace),math=mathLinks(trace,t,1,'v',{rowRange:[0,0],colRange:[0,0]});
  assert.deepEqual(math.operands.find(op=>op.key==='wv').range,{r0:0,r1:0,c0:0,c1:575});
  assert.match(math.formula,/F16 cache/);
});
test('cell and whole-matrix selections expose upstream inputs only',()=>{
  const trace=withWeights(),t=layout(trace);
  for(const key of ['wk','wv','wq','x'])for(const whole of [false,true]){
    const b=t.columns[0].blocks[key],item={rowRange:whole?[0,b.plan.rows-1]:[0,0],colRange:whole?[0,b.plan.cols-1]:[0,0]};
    assert.deepEqual(mathLinks(trace,t,0,key,item).operands,[]);
  }
  for(const kind of ['q','k','v']){
    const links=mathLinks(trace,t,1,kind,{rowRange:[0,0],colRange:[0,0]});
    assert.deepEqual([...new Set(links.operands.map(op=>op.key))].sort(),['w'+kind,'x'].sort());
  }
  const raw=fixture(8,2),withoutWeights=layout(raw);
  for(const key of ['q','k','v'])assert.deepEqual(mathLinks(raw,withoutWeights,1,key,{rowRange:[0,0],colRange:[0,0]}).operands,[]);
});
test('littleLLAMA-style routes are orthogonal and avoid unrelated matrix interiors',()=>{
  const obstacle={x:40,y:-10,w:20,h:20},points=routeConnection({x:0,y:0},{x:100,y:0},[obstacle]);
  assert.ok(points);assert.deepEqual(points[0],{x:0,y:0});assert.deepEqual(points.at(-1),{x:100,y:0});
  for(let i=1;i<points.length;i++){
    const a=points[i-1],b=points[i];assert.ok(a.x===b.x||a.y===b.y);
    if(a.y===b.y&&a.y>-10&&a.y<10)assert.ok(Math.max(a.x,b.x)<=40||Math.min(a.x,b.x)>=60);
    if(a.x===b.x&&a.x>40&&a.x<60)assert.ok(Math.max(a.y,b.y)<=-10||Math.min(a.y,b.y)>=10);
  }
});

test('compact cache boundaries stay after prefill across multiple decodes',()=>{
  for(const decodes of [1,2,8]){
    const trace=fixture(9,decodes),t=layout(trace,{compact:true});
    for(const key of ['kt','v_operand']){
      const b=t.columns[0].blocks[key];
      assert.equal(b.appendStart,9);
      const indices=b.appendAxis==='row'?b.plan.rowIndices:b.plan.colIndices;
      assert.ok(indices.indexOf(b.appendStart)>0);
    }
  }
});
test('compact decode columns reserve only output width and a small gutter',()=>{
  const t=layout(fixture(),{compact:true});
  for(const col of t.columns.slice(1)){
    const width=Math.max(...Object.values(col.blocks).map(b=>b.w));
    assert.equal(col.right,0);
    assert.equal(col.width,width+30);
    for(const b of Object.values(col.blocks))assert.equal(b.x,col.x);
  }
});

test('weights and K/V/Q use the same displayed rows, cell heights and boundaries',()=>{
  for(const prefix of [1,4,9,32]){
    const t=layout(withWeights(fixture(prefix,2)),{compact:true});
    for(const kind of ['k','v','q']){
      const weight=t.columns[0].blocks['w'+kind];
      for(const col of t.columns){
        const output=col.blocks[kind];
        assert.deepEqual(weight.plan.rowIndices,output.plan.rowIndices);
        assert.deepEqual(weight.plan.rowGap,output.plan.rowGap);
        assert.equal(weight.plan.cellHeight,output.plan.cellHeight);
        assert.equal(weight.h,output.h);
        assert.equal(weight.y,output.y);
      }
      for(const r of weight.plan.rowIndices)for(const c of weight.plan.colIndices)
        if(r!==null&&c!==null)assert.ok(Number.isFinite(weight.values[r][c]));
    }
  }
});

test('diamond omissions retain real outer corner cells without enlarging the proxy',()=>{
  const p=plan(64,576),rows=p.rowIndices.length,cols=p.colIndices.length;
  assert.equal(p.cellWidth,38);assert.equal(p.cellHeight,24);
  assert.equal(p.width,25+cols*38+1);assert.equal(p.height,19+rows*24+1);
  for(const [r,c] of [[0,0],[0,cols-1],[rows-1,0],[rows-1,cols-1]])assert.equal(cellAt(p,r,c).omitted,false);
  const center=cellAt(p,1,Math.floor(cols/2)-1);
  assert.equal(center.diamondHidden,true);assert.equal(center.omitted,true);
  assert.equal(hitTest(p,p.padX+(Math.floor(cols/2)-.5)*p.cellWidth,p.padY+1.5*p.cellHeight).omitted,true);
  assert.equal(regions(p,{r0:center.row,r1:center.row,c0:center.col,c1:center.col})[0].hidden,true);
});

test('single-axis omissions are straight bands with uniform fading across the complete axis',()=>{
  for(const [rows,cols] of [[64,4],[4,64]]){
    const p=plan(rows,cols),nr=p.rowIndices.length,nc=p.colIndices.length;
    for(let r=0;r<nr;r++)for(let c=0;c<nc;c++){
      const item=cellAt(p,r,c);
      assert.equal(item.diamond,null);assert.equal(item.diamondHidden,false);
      assert.equal(item.omitted,item.row===null||item.col===null);
      const sameBand=cellAt(p,rows===64?r:0,rows===64?0:c);
      assert.equal(item.fade,sameBand.fade);
    }
    assert.equal(cellAt(p,0,0).fade,1);
    assert.ok(cellAt(p,rows===64?5:0,rows===64?0:5).fade<.3);
    const extended=extendPlan(p,rows,cols+2);
    for(let r=0;r<nr;r++)for(let c=0;c<nc;c++)assert.deepEqual(cellAt(extended,r,c),cellAt(p,r,c));
  }
});

test('two-axis diamond vertices touch all four edges of the fixed prefill footprint',()=>{
  for(const [rows,cols] of [[64,576],[32,64]]){
    const p=plan(rows,cols),d=cellAt(p,1,1).diamond;
    assert.equal(d.cx-d.rx,0);assert.equal(d.cy-d.ry,0);
    assert.equal(d.cx+d.rx,p.colIndices.length);assert.equal(d.cy+d.ry,p.rowIndices.length);
  }
});

test('pinched star reveals more corner cells than a straight diamond with identical tips',()=>{
  for(const [rows,cols] of [[64,576],[32,64]]){
    const p=plan(rows,cols),nr=p.rowIndices.length,nc=p.colIndices.length;let restored=0;
    for(let r=0;r<nr;r++)for(let c=0;c<nc;c++){
      const item=cellAt(p,r,c);if(item.row===null||item.col===null)continue;
      const oldHidden=Math.abs(c+.5-nc/2)/(nc/2)+Math.abs(r+.5-nr/2)/(nr/2)<1;
      if(item.diamondHidden)assert.equal(oldHidden,true);
      if(oldHidden&&!item.diamondHidden)restored++;
    }
    assert.ok(restored>0);
  }
});

test('all token axes share Q sampling across projections, attention and appended caches',()=>{
  for(const prefix of [4,6,8,16,32,64])for(const compact of [true,false]){
    const trace=fixture(prefix,3),t=layout(trace,{compact}),first=t.columns[0].blocks,q=first.q.plan;
    for(const key of ['x','k','v','va','scores','a']){
      assert.deepEqual(first[key].plan.colIndices,q.colIndices);
      assert.deepEqual(first[key].plan.colGap,q.colGap);
    }
    for(const col of t.columns){
      const keys=[...q.colIndices,...Array.from({length:col.step.start+col.step.count-prefix},(_,i)=>prefix+i)];
      for(const key of ['a','scores']){
        assert.deepEqual(col.blocks[key].plan.rowIndices,keys);
        assert.deepEqual(col.blocks[key].plan.rowGap,q.colGap);
        if(col.index)assert.deepEqual(col.blocks[key].plan.colIndices,[0]);
      }
    }
    const cacheTokens=[...q.colIndices,...(compact?[prefix,prefix+1,prefix+2]:[])];
    assert.deepEqual(first.kt.plan.rowIndices,cacheTokens);
    assert.deepEqual(first.v_operand.plan.colIndices,cacheTokens);
  }
});

test('curved connectors preserve endpoints, smooth routed corners and clear obstacles',()=>{
  const obstacle={x:40,y:-10,w:20,h:20};
  const points=routeConnection({x:0,y:0},{x:100,y:0},[obstacle]),path=curvedConnection(points,[obstacle]);
  assert.ok(path.startsWith('M0,0'));assert.ok(path.endsWith('L100,0'));assert.match(path,/Q/);
  const matches=[...path.matchAll(/L([-\d.]+),([-\d.]+) Q([-\d.]+),([-\d.]+) ([-\d.]+),([-\d.]+)/g)];
  assert.ok(matches.length>0);
  for(const match of matches){
    const [x0,y0,x1,y1,x2,y2]=match.slice(1).map(Number);
    for(let i=0;i<=100;i++){const t=i/100,x=(1-t)**2*x0+2*(1-t)*t*x1+t*t*x2,y=(1-t)**2*y0+2*(1-t)*t*y1+t*t*y2;
      assert.ok(!(x>obstacle.x+.1&&x<obstacle.x+obstacle.w-.1&&y>obstacle.y+.1&&y<obstacle.y+obstacle.h-.1));
    }
  }
  assert.equal(curvedConnection([{x:0,y:0},{x:10,y:0},{x:20,y:0}]),'M0,0 L20,0');
});

test('range ribbons span complete endpoint faces and retain curved closed boundaries',()=>{
  const sideways=ribbonConnection({x:10,y:50},{x:200,y:100},20,80);
  assert.ok(sideways.startsWith('M10,40 C'));assert.ok(sideways.includes('200,60 L200,140 C'));assert.ok(sideways.endsWith('10,60 Z'));
  const vertical=ribbonConnection({x:50,y:10},{x:100,y:200},20,80,true);
  assert.ok(vertical.startsWith('M40,10 C'));assert.ok(vertical.includes('60,200 L140,200 C'));assert.ok(vertical.endsWith('60,10 Z'));
  const input=[{x:0,y:0,w:20,h:10,hidden:false},{x:0,y:10,w:20,h:10,hidden:false},{x:20,y:0,w:10,h:20,hidden:true}];
  const merged=mergeRegions(input);assert.equal(merged.length,2);assert.equal(merged[0].h,20);assert.equal(input[0].h,10);assert.equal(merged[1].hidden,true);
});

test('ribbons attach to exact selected-cell edges for operands on every side',()=>{
  const source={x:100,y:100,w:80,h:100},selected={x:120,y:120,w:20,h:20};
  for(const [target,start,end,vertical] of [
    [{x:220,y:100,w:60,h:100},{x:140,y:130},{x:220,y:150},false],
    [{x:0,y:100,w:60,h:100},{x:120,y:130},{x:60,y:150},false],
    [{x:100,y:0,w:80,h:60},{x:130,y:120},{x:140,y:60},true],
    [{x:100,y:240,w:80,h:60},{x:130,y:140},{x:140,y:240},true]
  ]){
    const result=connectionAnchors(source,target,selected,target);
    assert.deepEqual(result.start,start);assert.deepEqual(result.end,end);assert.equal(result.vertical,vertical);
  }
});

test('lifted tape curves rise from the plane and land on the same range endpoints',()=>{
  const path=ribbonConnection({x:0,y:100},{x:200,y:100},20,40,false,50);
  assert.ok(path.startsWith('M0,90 C76,40'));assert.ok(path.includes('200,80 L200,120 C'));assert.ok(path.endsWith('0,110 Z'));
  const vertical=ribbonConnection({x:100,y:0},{x:100,y:200},20,40,true,50);
  assert.ok(vertical.startsWith('M90,0 C140,76'));assert.ok(vertical.includes('80,200 L120,200 C'));assert.ok(vertical.endsWith('110,0 Z'));
});

test('ribbon lands on the contributing range rather than its matrix boundary',()=>{
  const result=connectionAnchors({x:100,y:100,w:80,h:100},{x:0,y:100,w:60,h:100},{x:150,y:160,w:20,h:20},{x:10,y:130,w:20,h:20});
  assert.deepEqual(result.start,{x:150,y:170});assert.deepEqual(result.end,{x:30,y:140});
  assert.equal(result.sourceWidth,20);assert.equal(result.targetWidth,20);
});

test('selection outlines leave omission transitions open instead of boxing each proxy slice',()=>{
  const rects=[{x:0,y:0,w:20,h:20},{x:20,y:0,w:10,h:20,hidden:true},{x:30,y:0,w:20,h:20}];
  assert.equal(rangeOutline(rects[0],rects),'M0,0h20M0,20h20M0,0v20');
  assert.equal(rangeOutline(rects[1],rects),'M20,0h10M20,20h10');
  assert.equal(rangeOutline(rects[2],rects),'M30,0h20M30,20h20M50,0v20');
  assert.equal(rangeOutline(rects[1],[rects[1]]),'M20,0h10M20,20h10M20,0v20M30,0v20');
  const partial=[rects[0],{x:20,y:5,w:10,h:10,hidden:true}];
  assert.ok(rangeOutline(partial[0],partial).endsWith('M20,0v5M20,15v5'));
});

test('selection outlines also leave horizontal omission transitions open',()=>{
  const rects=[{x:0,y:0,w:20,h:20},{x:0,y:20,w:20,h:10,hidden:true},{x:0,y:30,w:20,h:20}];
  const outlines=rects.map(r=>rangeOutline(r,rects));
  assert.ok(outlines[0].startsWith('M0,0h20'));assert.ok(!outlines[0].includes('M0,20h20'));
  assert.equal(outlines[1],'M0,20v10M20,20v10');
  assert.ok(outlines[2].startsWith('M0,50h20'));assert.ok(!outlines[2].includes('M0,30h20'));
});

(function(root){
  'use strict';
  const PAD_X=25,PAD_Y=19,OMISSION_POWER=.6;
  function omissionFade(distance){
    const t=Math.min(1,Math.max(0,(distance-1)/(Math.pow(1.60,OMISSION_POWER)-1)));
    return t*t*(3-2*t);
  }
  function axis(size,edge){
    if(size<=2*edge+1)return Array.from({length:size},(_,i)=>i);
    // Two display slots represent one omitted original-index range.
    return [...Array.from({length:edge},(_,i)=>i),null,null,...Array.from({length:edge},(_,i)=>size-edge+i)];
  }
  function plan(rows,cols){
    if(!Number.isInteger(rows)||!Number.isInteger(cols)||rows<1||cols<1)throw Error('Invalid matrix dimensions');
    const rowBudget=Math.max(2,Math.min(6,Math.round(8*Math.sqrt(rows)/(Math.sqrt(rows)+Math.sqrt(cols)))));
    const edgeRows=Math.min(4,rowBudget),edgeCols=Math.min(4,8-rowBudget);
    const rowIndices=axis(rows,edgeRows),colIndices=axis(cols,edgeCols);
    const cellWidth=38,cellHeight=24;
    const rowGap=rowIndices.includes(null)?[edgeRows,rows-edgeRows-1]:null;
    const colGap=colIndices.includes(null)?[edgeCols,cols-edgeCols-1]:null;
    return {rows,cols,edgeRows,edgeCols,rowIndices,colIndices,rowGap,colGap,cellWidth,cellHeight,
      padX:PAD_X,padY:PAD_Y,width:PAD_X+colIndices.length*cellWidth+1,height:PAD_Y+rowIndices.length*cellHeight+1,
      retained:rowIndices.filter(x=>x!==null).length*colIndices.filter(x=>x!==null).length};
  }
  function extendPlan(base,rows,cols,{rowStart=0,colStart=0}={}){
    const extend=(indices,initial,size,start)=>start?Array.from({length:size-start},(_,i)=>start+i):[...indices,...Array.from({length:size-initial},(_,i)=>initial+i)];
    const rowIndices=extend(base.rowIndices,base.rows,rows,rowStart),colIndices=extend(base.colIndices,base.cols,cols,colStart);
    return {...base,rows,cols,rowIndices,colIndices,fadeRows:base.rows,fadeCols:base.cols,
      rowGap:rowStart?null:base.rowGap,colGap:colStart?null:base.colGap,
      width:base.padX+colIndices.length*base.cellWidth+1,height:base.padY+rowIndices.length*base.cellHeight+1,
      retained:rowIndices.filter(x=>x!==null).length*colIndices.filter(x=>x!==null).length};
  }
  function cellAt(p,ri,ci){
    const row=p.rowIndices[ri],col=p.colIndices[ci];
    if(row===undefined||col===undefined)return null;
    const rowRange=row===null?p.rowGap:[row,row],colRange=col===null?p.colGap:[col,col];
    const dr=p.rowGap&&row!==null?Math.max(0,Math.min(row,(p.fadeRows||p.rows)-1-row))/Math.max(1,p.edgeRows-1):0;
    const dc=p.colGap&&col!==null?Math.max(0,Math.min(col,(p.fadeCols||p.cols)-1-col))/Math.max(1,p.edgeCols-1):0;
    const axes=Number(Boolean(p.rowGap))+Number(Boolean(p.colGap));
    const inward=axes?(dr+dc)/axes:0;
    // Use the original prefill footprint so appending does not move the hole.
    const nr=p.rowIndices.length-Math.max(0,p.rows-(p.fadeRows||p.rows));
    const nc=p.colIndices.length-Math.max(0,p.cols-(p.fadeCols||p.cols));
    // A diamond is meaningful only when both dimensions are compressed.
    // One-axis omissions remain straight bands, fading only along that axis.
    const diamond=Boolean(p.rowGap&&p.colGap)&&nr>2&&nc>2&&ri<nr&&ci<nc;
    const rx=nc/2,ry=nr/2;
    const radius=diamond?Math.pow(Math.abs(ci+.5-nc/2)/rx,OMISSION_POWER)+Math.pow(Math.abs(ri+.5-nr/2)/ry,OMISSION_POWER):Infinity;
    const diamondHidden=radius<1;
    const adjacentRowGap=p.rowGap&&(ri===p.rowIndices.indexOf(null)-1||ri===p.rowIndices.lastIndexOf(null)+1);
    const adjacentColGap=p.colGap&&(ci===p.colIndices.indexOf(null)-1||ci===p.colIndices.lastIndexOf(null)+1);
    const gapFade=(adjacentRowGap?.85:1)*(adjacentColGap?.85:1);
    const fade=(diamond?omissionFade(radius):1-.82*Math.pow(inward,1.3))*gapFade;
    return {row,col,rowRange,colRange,omitted:row===null||col===null||diamondHidden,diamondHidden,diamond:diamond?{cx:nc/2,cy:nr/2,rx,ry}:null,fade};
  }
  function hitTest(p,x,y){return cellAt(p,Math.floor((y-p.padY)/p.cellHeight),Math.floor((x-p.padX)/p.cellWidth));}
  function matrixRange(p){
    return {rowRange:[p.rowIndices[0],p.rowIndices.at(-1)],colRange:[p.colIndices[0],p.colIndices.at(-1)]};
  }
  function formatNumber(value){
    if(!Number.isFinite(value))return '—';
    return (Math.abs(value)<.005?0:value).toFixed(2);
  }
  function weightValues(weight){
    const rows=Array.from({length:weight.rows},()=>new Array(weight.cols));
    weight.row_indices.forEach((r,i)=>weight.col_indices.forEach((c,j)=>{rows[r][c]=weight.values[i][j];}));
    return rows;
  }
  function scoreValues(step){
    const m=step.matrices;
    if(m.scores)return m.scores;
    return m.kt.map(key=>Array.from({length:m.q[0].length},(_,c)=>key.reduce((sum,v,i)=>sum+v*m.q[i][c],0)/Math.sqrt(m.q.length)));
  }
  function layout(trace,{compact=false,showProjections=true}={}){
    const last=trace.steps.length-1,latest=trace.steps[last],prefill=trace.steps[0];
    const tokens=plan(prefill.matrices.q.length,prefill.count);
    const columns=trace.steps.map((step,index)=>{
      const blocks={};
      function add(key,values,options={}){
        const originalKey=key==='v_operand'?'vc':key,initial=key.startsWith('w')?values:(key==='scores'?scoreValues(prefill):prefill.matrices[originalKey]);
        let base=plan(initial.length,initial[0].length),p;
        // Token positions are one shared axis, independent of each matrix's
        // other dimension. Attention and cached operands must match Q.
        if(['x','k','v','q','va','a','scores','v_operand'].includes(key))
          base={...base,colIndices:tokens.colIndices,colGap:tokens.colGap,edgeCols:tokens.edgeCols};
        if(['kt','a','scores'].includes(key))
          base={...base,rowIndices:tokens.colIndices,rowGap:tokens.colGap,edgeRows:tokens.edgeCols};
        if(key.startsWith('w')){
          const output=blocks[key.slice(1)].plan;
          base={...base,rowIndices:output.rowIndices,rowGap:output.rowGap,edgeRows:output.edgeRows,cellHeight:output.cellHeight};
        }
        if(index&&['x','k','v','q','va','a','scores'].includes(key)){
          const narrow=plan(initial.length,values[0].length);
          base={...base,cols:narrow.cols,colIndices:narrow.colIndices,colGap:narrow.colGap,cellWidth:narrow.cellWidth};
        }
        if(key==='x')base={...base,padY:base.padY+48};
        const suffix=!compact&&index&&['kt','v_operand','kc','vc'].includes(key);
        p=extendPlan(base,values.length,values[0].length,suffix?(key==='kt'?{rowStart:step.start}:{colStart:step.start}):{});
        blocks[key]={key,kind:key==='v_operand'?'vc':key,values,appendOnly:Boolean(suffix),plan:p,w:p.width,h:p.height,index,sourceStep:index,...options};
      }
      add('scores',scoreValues(step));
      for(const key of showProjections?['x','k','v','q','a','va']:['x','q','a','va'])add(key,step.matrices[key],{causal:key==='a'&&index===0});
      if(showProjections&&index===0&&trace.weights)for(const key of ['wk','wv','wq'])add(key,weightValues(trace.weights[key]),{parameter:true,shared:true});
      if(!compact||index===0){
        const source=compact?latest:step,sourceStep=compact?last:index;
        const appendStart=compact&&last>0?prefill.count:step.start;
        add('kt',source.matrices.kt,{sourceStep,shared:compact,appendAxis:'row',appendStart:appendStart||null});
        add('v_operand',source.matrices.vc,{sourceStep,shared:compact,appendAxis:'col',appendStart:appendStart||null});
      }
      return {index,step,blocks};
    });
    const rowHeight=keys=>Math.max(...columns.flatMap(c=>keys.map(key=>c.blocks[key]?.h||0)));
    const y={x:80};
    y.k=y.x+rowHeight(['x'])+12;y.v=y.k+rowHeight(['k','wk'])+12;y.q=y.v+rowHeight(['v','wv'])+12;
    if(!showProjections)y.q=y.x+rowHeight(['x'])+12;
    y.scores=y.q+rowHeight(['q','wq'])+12;
    y.a=y.scores+rowHeight(['scores','kt'])+12;y.va=y.a+rowHeight(['a'])+12;
    const divider=y.va+rowHeight(['va','v_operand'])+24;
    const height=divider+20;
    let left=185;
    for(const col of columns){
      const b=col.blocks,operand=Math.max(...['kt','v_operand','wk','wv','wq'].map(key=>{const block=b[key];if(!block)return 0;if(compact&&['kt','v_operand'].includes(key))return plan(prefill.matrices[key==='kt'?'kt':'vc'].length,prefill.matrices[key==='kt'?'kt':'vc'][0].length).width;return block.w;}));
      const right=operand?operand+24:0;
      const output=Math.max(...['x','k','v','q','scores','a','va'].map(key=>b[key]?.w||0));
      col.x=left;col.right=right;col.width=right+output+30;col.height=height;
      for(const block of Object.values(b)){
        block.x=left+(['kt','v_operand','wk','wv','wq'].includes(block.key)?operand-block.w:right);
        block.y=y[block.key==='kt'?'scores':block.key==='v_operand'?'va':block.key.startsWith('w')?block.key.slice(1):block.key];
        if(block.key==='kt'&&block.appendOnly){const row=b.scores.plan.rowIndices.indexOf(col.step.start);block.y+=row*b.scores.plan.cellHeight;} 
      }
      left+=col.width;
    }
    return {columns,y,divider,width:left+15,height,groups:[
      {title:'Input',top:y.x,height:rowHeight(['x']),labels:[],centers:[]},
      {title:'Projections',top:showProjections?y.k:y.q,height:y.scores-(showProjections?y.k:y.q)-12,labels:[],centers:[]},
      {title:'Attention',top:y.scores,height:y.va-y.scores-12,labels:[],centers:[]},
      {title:'Value aggregation',top:y.va,height:rowHeight(['va','v_operand']),labels:[],centers:[]}
    ]};
  }
  // Intersect original index ranges with proxy cells, keeping omitted spans explicit.
  function regions(p,range){
    function runs(indices,gap,start,end){
      const out=[];
      indices.forEach((value,i)=>{
        const bounds=value===null?gap:[value,value];
        if(bounds[1]<start||bounds[0]>end)return;
        const hidden=value===null,last=out.at(-1);
        if(last&&last.hidden===hidden&&last.end===i-1)last.end=i;
        else out.push({start:i,end:i,hidden});
      });return out;
    }
    if(p.rowGap&&p.colGap){
      const out=[];
      for(let ri=0;ri<p.rowIndices.length;ri++){
        let last=null;
        for(let ci=0;ci<p.colIndices.length;ci++){
          const item=cellAt(p,ri,ci);
          if(item.rowRange[1]<range.r0||item.rowRange[0]>range.r1||item.colRange[1]<range.c0||item.colRange[0]>range.c1)continue;
          if(last&&last.hidden===item.omitted&&last.end===ci-1){last.w+=p.cellWidth;last.end=ci;}
          else{last={x:p.padX+ci*p.cellWidth,y:p.padY+ri*p.cellHeight,w:p.cellWidth,h:p.cellHeight,hidden:item.omitted,end:ci};out.push(last);}
        }
      }
      return out.map(({end,...rect})=>rect);
    }
    return runs(p.rowIndices,p.rowGap,range.r0,range.r1).flatMap(r=>runs(p.colIndices,p.colGap,range.c0,range.c1).map(c=>({
      x:p.padX+c.start*p.cellWidth,y:p.padY+r.start*p.cellHeight,
      w:(c.end-c.start+1)*p.cellWidth,h:(r.end-r.start+1)*p.cellHeight,hidden:r.hidden||c.hidden
    })));
  }
  function mathLinks(trace,timeline,index,key,item){
    const step=trace.steps[index],m=step.matrices,operands=[],range={r0:item.rowRange[0],r1:item.rowRange[1],c0:item.colRange[0],c1:item.colRange[1]};
    const limit=index?step.start+step.count-1:Math.min(step.count-1,range.c1);
    function add(i,k,r,role){if(timeline.columns[i]?.blocks[k])operands.push({index:i,key:k,range:r,role});}
    function caches(k,r,role){
      if(timeline.columns[0].blocks[k]?.shared){add(0,k,r,role);return;}
      for(const col of timeline.columns){if(col.index>index)break;add(col.index,k,r,role);}
    }
    let formula='',note='Solid connections show displayed operands; dashed connections pass through omitted ranges.';
    function projection(kind){
      const w=trace.weights?.['w'+kind];if(!w)return false;
      let rows=[];
      for(let r=range.r0;r<=range.r1;r++){
        rows.push(r);
        const rot=trace.weights.rope_dims||m[kind].length;
        if(kind!=='v'&&r<rot&&trace.weights.rope_type!==-1){
          const pair=trace.weights.rope_type===2?(r<rot/2?r+rot/2:r-rot/2):(r^1);rows.push(pair);
        }
      }
      if(kind!=='v'&&trace.weights.head_norm)rows=Array.from({length:m[kind].length},(_,i)=>i);
      rows=[...new Set(rows)].sort((a,b)=>a-b);
      let start=rows[0],end=start;
      const emit=()=>add(0,'w'+kind,{r0:start,r1:end,c0:0,c1:w.cols-1},'Weight row'+(kind==='v'?'':' and RoPE partner'));
      for(const r of rows.slice(1)){if(r===end+1)end=r;else{emit();start=end=r;}}emit();
      add(index,'x',{r0:0,r1:w.cols-1,c0:range.c0,c1:range.c1},'Input column');
      const transform=trace.weights.transforms?.[kind]||'linear';
      formula=`Z${kind} = W${kind} × X; ${kind.toUpperCase()}: ${transform}`;
      if(!timeline.columns[0].blocks['w'+kind])note+=' Expand Weights and K/V to display the weight operand.';
      note+=' Weights are dequantized GGUF values. RoPE mixes paired components; runtime quantization and cache writes can round the result.';
      return true;
    }
    if(['q','k','v'].includes(key)&&projection(key))return {formula,note,operands:operands.filter(op=>regions(timeline.columns[op.index].blocks[op.key].plan,op.range).length)};
    if(['q','k','v'].includes(key))return {formula:'',note:'The projection inputs are not available in this trace.',operands:[]};
    if(key.startsWith('w')){
      return {formula:'',note:'Model parameters loaded from GGUF; no upstream matrix is displayed for these weights.',operands:[]};
    }
    if(key==='x'){
      return {formula:'',note:'Normalized attention input; its upstream computation is outside this view.',operands:[]};
    }
    if(key==='a'&&index===0&&range.r0>range.c1)return {formula:'A[t, j] = 0 (causal mask)',note:'Future tokens do not participate in this attention calculation.',operands:[]};
    if(key==='scores'){
      formula='Scores[t, j] = Σd Kᵀ[t, d] × Q[d, j] / √dₖ';
      add(index,'q',{r0:0,r1:m.q.length-1,c0:range.c0,c1:range.c1},'Query column');
      caches('kt',{r0:range.r0,r1:range.r1,c0:0,c1:m.q.length-1},'Key row');
    }else if(key==='a'){
      formula='S = KᵀQ / √dₖ + causal mask; A[:, j] = softmax(S[:, j])';
      add(index,'scores',{r0:0,r1:limit,c0:range.c0,c1:range.c1},'Scores for column-wise softmax');
      note+=' Softmax uses all causally visible scores in this query column.';
    }else if(key==='va'){
      formula='VA[d, j] = Σₜ V[d, t] × A[t, j]';
      caches('v_operand',{r0:range.r0,r1:range.r1,c0:0,c1:limit},'Value row');
      add(index,'a',{r0:0,r1:limit,c0:range.c0,c1:range.c1},'Attention column');
    }else if(key==='kt'||key==='kc'){
      formula='Kᵀ[t, d] = K[d, t]';
      for(const col of timeline.columns){const start=Math.max(range.r0,col.step.start),end=Math.min(range.r1,col.step.start+col.step.count-1);if(start<=end)add(col.index,'k',{r0:range.c0,r1:range.c1,c0:start-col.step.start,c1:end-col.step.start},'Cached key projection, transposed');}
    }else if(key==='v_operand'||key==='vc'){
      formula='V cached[d, t] = V[d, t]';
      for(const col of timeline.columns){const start=Math.max(range.c0,col.step.start),end=Math.min(range.c1,col.step.start+col.step.count-1);if(start<=end)add(col.index,'v',{r0:range.r0,r1:range.r1,c0:start-col.step.start,c1:end-col.step.start},'Cached value projection');}
    }else note='Projection weights are not displayed in this view.';
    return {formula,note,operands:operands.filter(op=>regions(timeline.columns[op.index].blocks[op.key].plan,op.range).length)};
  }
  // As in littleLLAMA, route blue orthogonal connectors through matrix gutters.
  function routeConnection(start,end,obstacles){
    const unique=values=>[...new Set(values.map(x=>Math.round(x*100)/100))].sort((a,b)=>a-b);
    const xs=unique([start.x,end.x,...obstacles.flatMap(r=>[r.x-8,r.x+r.w+8])]),ys=unique([start.y,end.y,...obstacles.flatMap(r=>[r.y-8,r.y+r.h+8])]);
    const nearest=(a,v)=>a.reduce((best,x,i)=>Math.abs(x-v)<Math.abs(a[best]-v)?i:best,0),key=(x,y)=>y*xs.length+x;
    const sx=nearest(xs,start.x),sy=nearest(ys,start.y),ex=nearest(xs,end.x),ey=nearest(ys,end.y),origin=key(sx,sy),goal=key(ex,ey);
    const cost=new Map([[origin,0]]),parents=new Map(),closed=new Set(),open=[{x:sx,y:sy,id:origin,f:0}];
    function blocked(a,b){return obstacles.some(r=>a.x===b.x?a.x>r.x+.1&&a.x<r.x+r.w-.1&&Math.max(a.y,b.y)>r.y+.1&&Math.min(a.y,b.y)<r.y+r.h-.1:a.y>r.y+.1&&a.y<r.y+r.h-.1&&Math.max(a.x,b.x)>r.x+.1&&Math.min(a.x,b.x)<r.x+r.w-.1);}
    while(open.length&&closed.size<20000){
      open.sort((a,b)=>b.f-a.f);const node=open.pop();if(closed.has(node.id))continue;
      if(node.id===goal){const points=[];for(let id=goal;id!==undefined;id=parents.get(id))points.push({x:xs[id%xs.length],y:ys[Math.floor(id/xs.length)]});return points.reverse();}
      closed.add(node.id);
      for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1]]){
        const x=node.x+dx,y=node.y+dy;if(x<0||y<0||x>=xs.length||y>=ys.length)continue;
        const id=key(x,y);if(closed.has(id))continue;
        const a={x:xs[node.x],y:ys[node.y]},b={x:xs[x],y:ys[y]};if(blocked(a,b))continue;
        const next=cost.get(node.id)+Math.abs(a.x-b.x)+Math.abs(a.y-b.y);if(next>=(cost.get(id)??Infinity))continue;
        cost.set(id,next);parents.set(id,node.id);open.push({x,y,id,f:next+Math.abs(b.x-xs[ex])+Math.abs(b.y-ys[ey])});
      }
    }return null;
  }
  function curvedConnection(points,obstacles=[],radius=24){
    if(!points?.length)return '';
    const line=[];
    for(const p of points){
      if(line.length&&line.at(-1).x===p.x&&line.at(-1).y===p.y)continue;
      if(line.length>1){const a=line.at(-2),b=line.at(-1);if((b.x-a.x)*(p.y-b.y)===(b.y-a.y)*(p.x-b.x))line.pop();}
      line.push(p);
    }
    let d=`M${line[0].x},${line[0].y}`;
    for(let i=1;i<line.length-1;i++){
      const a=line[i-1],b=line[i],c=line[i+1],ab=Math.hypot(b.x-a.x,b.y-a.y),bc=Math.hypot(c.x-b.x,c.y-b.y);
      let r=Math.min(radius,ab/2,bc/2),entry,exit;
      for(let attempt=0;attempt<8;attempt++){
        entry={x:b.x+(a.x-b.x)*r/ab,y:b.y+(a.y-b.y)*r/ab};
        exit={x:b.x+(c.x-b.x)*r/bc,y:b.y+(c.y-b.y)*r/bc};
        const collision=Array.from({length:25},(_,j)=>j/24).some(t=>{
          const x=(1-t)**2*entry.x+2*(1-t)*t*b.x+t*t*exit.x,y=(1-t)**2*entry.y+2*(1-t)*t*b.y+t*t*exit.y;
          return obstacles.some(o=>x>o.x+.1&&x<o.x+o.w-.1&&y>o.y+.1&&y<o.y+o.h-.1);
        });
        if(!collision)break;
        r/=2;
      }
      d+=` L${entry.x},${entry.y} Q${b.x},${b.y} ${exit.x},${exit.y}`;
    }
    if(line.length>1)d+=` L${line.at(-1).x},${line.at(-1).y}`;
    return d;
  }
  function connectionAnchors(source,target,selected,range){
    const above=target.y+target.h<=source.y,below=target.y>=source.y+source.h;
    if(above||below)return {
      start:{x:selected.x+selected.w/2,y:above?selected.y:selected.y+selected.h},
      end:{x:range.x+range.w/2,y:above?range.y+range.h:range.y},
      sourceWidth:selected.w,targetWidth:range.w,vertical:true
    };
    const toLeft=target.x+target.w/2<source.x+source.w/2;
    return {
      start:{x:toLeft?selected.x:selected.x+selected.w,y:selected.y+selected.h/2},
      end:{x:toLeft?range.x+range.w:range.x,y:range.y+range.h/2},
      sourceWidth:selected.h,targetWidth:range.h,vertical:false
    };
  }
  function ribbonConnection(start,end,startWidth,endWidth,vertical=false,lift=0){
    const a=startWidth/2,b=endWidth/2;
    if(lift){
      if(vertical){
        const dy=(end.y-start.y)*.38;
        return `M${start.x-a},${start.y} C${start.x-a+lift},${start.y+dy} ${end.x-b+lift},${end.y-dy} ${end.x-b},${end.y} L${end.x+b},${end.y} C${end.x+b+lift},${end.y-dy} ${start.x+a+lift},${start.y+dy} ${start.x+a},${start.y} Z`;
      }
      const dx=(end.x-start.x)*.38;
      return `M${start.x},${start.y-a} C${start.x+dx},${start.y-a-lift} ${end.x-dx},${end.y-b-lift} ${end.x},${end.y-b} L${end.x},${end.y+b} C${end.x-dx},${end.y+b-lift} ${start.x+dx},${start.y+a-lift} ${start.x},${start.y+a} Z`;
    }
    if(vertical){
      const mid=(start.y+end.y)/2;
      return `M${start.x-a},${start.y} C${start.x-a},${mid} ${end.x-b},${mid} ${end.x-b},${end.y} L${end.x+b},${end.y} C${end.x+b},${mid} ${start.x+a},${mid} ${start.x+a},${start.y} Z`;
    }
    const mid=(start.x+end.x)/2;
    return `M${start.x},${start.y-a} C${mid},${start.y-a} ${mid},${end.y-b} ${end.x},${end.y-b} L${end.x},${end.y+b} C${mid},${end.y+b} ${mid},${start.y+a} ${start.x},${start.y+a} Z`;
  }
  function mergeRegions(rects){
    const merged=[];
    for(const rect of rects){
      const previous=merged.find(r=>r.x===rect.x&&r.w===rect.w&&r.hidden===rect.hidden&&Math.abs(r.y+r.h-rect.y)<.001);
      if(previous)previous.h+=rect.h;else merged.push({...rect});
    }
    return merged;
  }
  function rangeOutline(rect,rects){
    let d='';
    for(const vertical of [false,true])for(const far of [false,true]){
      const coordinate=(vertical?rect.x:rect.y)+(far?(vertical?rect.w:rect.h):0);
      const low=vertical?rect.y:rect.x,high=low+(vertical?rect.h:rect.w);
      let spans=[[low,high]];
      for(const other of rects){
        const opposite=(vertical?other.x:other.y)+(far?0:(vertical?other.w:other.h));
        if(other===rect||Math.abs(opposite-coordinate)>.001)continue;
        const a=vertical?other.y:other.x,b=a+(vertical?other.h:other.w);
        spans=spans.flatMap(([start,end])=>b<=start||a>=end?[[start,end]]:[...(a>start?[[start,Math.min(a,end)]]:[]),...(b<end?[[Math.max(b,start),end]]:[])]);
      }
      for(const [a,b] of spans)if(b>a)d+=vertical?`M${coordinate},${a}v${b-a}`:`M${a},${coordinate}h${b-a}`;
    }
    return d;
  }
  const api={OMISSION_POWER,omissionFade,scoreValues,axis,plan,extendPlan,cellAt,hitTest,matrixRange,formatNumber,layout,regions,mathLinks,weightValues,routeConnection,curvedConnection,ribbonConnection,mergeRegions,connectionAnchors,rangeOutline};
  if(typeof module!=='undefined')module.exports=api;
  root.KavinProxy=api;
})(typeof window==='undefined'?globalThis:window);

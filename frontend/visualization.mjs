import {scaleLinear, select, zoomIdentity, easeCubicInOut} from 'd3';
const ramps=new Map();
export function colorRamp(neutral,base,intensity){
  const key=neutral.join(',')+':'+base.join(',');
  let ramp=ramps.get(key);
  if(!ramp){ramp=scaleLinear().domain([0,1]).range([neutral,base]).interpolate((a,b)=>t=>a.map((value,i)=>value+(b[i]-value)*t)).clamp(true);ramps.set(key,ramp);}
  return `rgb(${ramp(intensity).map(Math.round).join(',')})`;
}
export const cameraTransform=({x,y,zoom})=>zoomIdentity.translate(x,y).scale(zoom);
export const cameraEase=easeCubicInOut;
export function applySceneTransform(element,camera){
  const transform=cameraTransform(camera);
  select(element).style('transform',`translate(${transform.x}px,${transform.y}px) scale(${transform.k})`)
    .style('--label-scale',1/transform.k).classed('overview',transform.k<.7-1e-6);
}
// SVG annotations and connector nodes are renderer-owned children of React's
// empty hosts. React never reconciles these nodes or the Canvas heatmap cells.
export function svgNode(tag,attrs={},content){
  const selection=select(document.createElementNS('http://www.w3.org/2000/svg',tag));
  for(const [key,value] of Object.entries(attrs))selection.attr(key,value);
  if(content!==undefined)selection.text(content);
  return selection.node();
}

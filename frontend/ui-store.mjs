// React owns application chrome. The Canvas/SVG renderer owns only its memoized
// hosts, so camera frames never trigger React reconciliation of matrix cells.
let snapshot = {
  ready:false,busy:false,playing:false,canStep:false,hasTrace:false,
  modelName:'Choose a model',layerCount:0,headCount:0,layer:0,head:0,kvHead:0,
  emptyTitle:'Choose a model',emptyDescription:'Load a model to inspect its KV cache.',
  emptyActionHidden:false,status:'Choose a model to begin',metrics:'LOCAL INFERENCE · NO TRAINING',
  modelLoading:false,loadLabel:'Load model',customModel:false,validation:'',
  progressVisible:false,progress:0,progressMessage:'',toast:'',demoLimit:false,
  localOnlyMessage:'',localOnlyOpen:false
};
const listeners = new Set();
export const uiStore = {
  getSnapshot:()=>snapshot,
  subscribe(listener){listeners.add(listener);return ()=>listeners.delete(listener);},
  set(patch){
    if(!Object.keys(patch).some(key=>snapshot[key]!==patch[key]))return;
    snapshot={...snapshot,...patch};
    for(const listener of listeners)listener();
  }
};
export const actions = {};

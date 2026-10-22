import test from 'node:test';
import assert from 'node:assert/strict';
import {cameraTransform,cameraEase,colorRamp} from '../frontend/visualization.mjs';
import {uiStore} from '../frontend/ui-store.mjs';

test('D3 color scales preserve the legacy heatmap colors exactly',()=>{
  const colors=[[138,219,139],[181,94,184],[135,145,155],[0,181,226],[241,215,83],[242,163,104]];
  for(const base of colors){
    for(const neutral of [base.map(v=>Math.round(255*.88+v*.12)),[244,230,247]]){
      for(let i=0;i<=1000;i++){
        const intensity=i/1000;
        const legacy=`rgb(${neutral.map((v,j)=>Math.round(v+(base[j]-v)*intensity)).join(',')})`;
        assert.equal(colorRamp(neutral,base,intensity),legacy);
      }
    }
  }
});

test('D3 transforms preserve pointer anchoring across the full camera zoom range',()=>{
  for(const zoom of [.025,.15,.7,1,1.5,3]){
    const state={x:-145.75,y:82.5,zoom},transform=cameraTransform(state);
    const point=[121.2,456.8],scene=transform.invert(point),restored=transform.apply(scene);
    for(let i=0;i<2;i++)assert.ok(Math.abs(restored[i]-point[i])<1e-10);
    const next=Math.min(3,zoom*1.25);
    assert.ok(Math.abs(point[0]-scene[0]*next-(point[0]-(point[0]-state.x)*next/zoom))<1e-10);
  }
});

test('D3 easing matches the existing camera animation curve',()=>{
  for(let i=0;i<=1000;i++){
    const t=i/1000,legacy=t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
    assert.ok(Math.abs(cameraEase(t)-legacy)<1e-12);
  }
});

test('React UI snapshots are stable on no-op updates and independent of camera frames',()=>{
  const before=uiStore.getSnapshot();let updates=0;
  const unsubscribe=uiStore.subscribe(()=>updates++);
  uiStore.set({busy:before.busy});assert.equal(updates,0);assert.equal(uiStore.getSnapshot(),before);
  for(let i=0;i<100;i++)cameraTransform({x:i,y:-i,zoom:.7});
  assert.equal(uiStore.getSnapshot(),before);assert.equal(updates,0);
  uiStore.set({busy:!before.busy});assert.equal(updates,1);assert.equal(before.busy,!uiStore.getSnapshot().busy);
  unsubscribe();uiStore.set({busy:before.busy});assert.equal(updates,1);
});

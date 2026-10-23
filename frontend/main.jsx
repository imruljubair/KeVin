import React, {memo, useLayoutEffect, useRef, useSyncExternalStore} from 'react';
import {createRoot} from 'react-dom/client';
import {uiStore, actions} from './ui-store.mjs';
import {startController} from './controller.mjs';
const demo = Boolean(window.KevinDemo);
const useUI=()=>useSyncExternalStore(uiStore.subscribe,uiStore.getSnapshot,uiStore.getSnapshot);
function Header(){const ui=useUI();return (<><header className="topbar">
    <a className="brand" href="./" aria-label="KeVin home"><img className="brand-symbol" src="./favicon.svg" alt="" aria-hidden="true" /><strong>KeVin<span>Key–Value Cache Inspection Tool</span></strong></a>
    <button id="model-button" className={"model-button"+(demo?" local-only":"")} disabled={ui.busy||ui.playing} onClick={()=>actions.openModel()} title={demo?"Local only · load another model":undefined} aria-description={demo?"Local only. Activate for installation instructions.":undefined}><span className={"status-dot"+(ui.ready?" ready":"")}></span><span id="model-name">{ui.modelName}</span><span className="chevron">⌄</span></button>
    <span className="local-badge">{demo?"RECORDED DEMO":"ON YOUR MACHINE"}</span>
    {demo&&<a className="run-local-link" href="run-locally.html">Run locally ↗</a>}
    <button id="about-button" onClick={()=>document.getElementById("about-dialog").showModal()} className="icon-button" aria-label="About this visualization">?</button>
  </header></>);}
function Controls(){const ui=useUI();return (<><section className="control-rail" aria-label="Inference controls">
    <form id="prompt-form" className="sequence-editor" onSubmit={event=>{event.preventDefault();actions.prefill();}}>
      <label htmlFor="prompt">{demo?"Prompt · local only":"Prompt"}</label>
      <input id="prompt" disabled={ui.busy||ui.playing} readOnly={demo} aria-readonly={demo||undefined} className={demo?"local-only":undefined} title={demo?"Fixed demo prompt · edit prompts in the local version":undefined} onClick={()=>{if(demo)window.KevinDemo.localOnly("Run KeVin locally to enter your own prefill text.");}} onKeyDown={event=>{if(demo&&((event.key.length===1&&!event.ctrlKey&&!event.metaKey)||["Backspace","Delete"].includes(event.key))){event.preventDefault();window.KevinDemo.localOnly("Run KeVin locally to enter your own prefill text.");}}} defaultValue="Once upon a time" autoComplete="off" maxLength="4000" placeholder="Enter a short prompt…" />
      <button id="prefill" className="primary" type="submit" disabled={!ui.ready||ui.busy||ui.playing} title={demo?"Replay the fixed recorded prefill from the beginning":undefined}>{demo?"Replay prefill":"Prefill"} <span>↵</span></button>
      <span className="separator"></span>
      <button id="step" type="button" disabled={ui.demoLimit?ui.busy:!ui.canStep||ui.busy||ui.playing} className={ui.demoLimit?"local-only":undefined} onClick={()=>actions.step()} title={demo?(ui.demoLimit?"Local only · run locally to generate more tokens":"Reveal the next recorded token"):undefined} aria-description={demo?(ui.demoLimit?"Local only. Activate for installation instructions.":"Recorded inference."):undefined}>{ui.demoLimit?"Generate more · local only":"Generate next token"} <span>→</span></button>
      <button id="play" type="button" className={"icon-button"+(ui.demoLimit?" local-only":"")} disabled={ui.demoLimit?ui.busy:ui.playing?false:!ui.canStep||ui.busy} aria-label={ui.playing?"Pause generation":"Play generation"} onClick={()=>actions.play()} aria-description={demo?(ui.demoLimit?"Local only. Activate for installation instructions.":"Recorded inference."):undefined} title={demo?(ui.demoLimit?"Local only · run locally to generate more tokens":"Play recorded tokens"):undefined}>{ui.playing?"Ⅱ":"▶"}</button>
    </form>
    <div className="view-controls">
      <label>Layer <select id="layer" aria-label="Select layer" disabled={!ui.ready||ui.busy||ui.playing} value={ui.layer} onChange={event=>{const value=Number(event.target.value);uiStore.set({layer:value});actions.view();}}>{ui.layerCount?Array.from({length:ui.layerCount},(_,i)=><option key={i} value={i}>{i}</option>):<option value={0}>—</option>}</select></label>
      <label>Query head <select id="head" aria-label="Select query head" disabled={!ui.ready||ui.busy||ui.playing} value={ui.head} onChange={event=>{const value=Number(event.target.value);uiStore.set({head:value});actions.view();}}>{ui.headCount?Array.from({length:ui.headCount},(_,i)=><option key={i} value={i}>{i}</option>):<option value={0}>—</option>}</select></label>
      <span id="kv-head" className="kv-head">KV {ui.ready?ui.kvHead:"—"}</span>
      <span className="separator"></span>
      <label className="check"><input id="compact" type="checkbox" defaultChecked onChange={()=>actions.compact()} />Compact cache</label>
      <label className="check"><input id="numbers" type="checkbox" defaultChecked onChange={()=>actions.numbers()} />Numbers</label>
      <label className="check" title="Enable cell and matrix calculation explanations"><input id="math-mode" type="checkbox" checked={ui.mathMode} onChange={event=>actions.mathMode(event.target.checked)} />Math</label>
    </div>
  </section></>);}
const Viewport=memo(function Viewport(){return (<><main id="viewport" tabIndex="0" aria-label="KV cache visualization. Drag to pan; use zoom controls to inspect matrices.">
    <EmptyState/>
    <div id="scene-clip"><div id="scene" className="scene" hidden><svg id="connections" aria-hidden="true"></svg><div id="scene-labels"></div><div id="columns"></div></div></div>
    <div id="axis-labels" hidden></div>
    <div id="step-labels" hidden></div><div id="decode-group" hidden></div>
  </main></>);});
// Camera-only text is an explicit imperative host, independent of UI snapshots.
const ScaleNote=memo(()=> <span id="scale-note" className="scale-note">Original corner cells · shaded gaps are omitted entries</span>);

function BottomRail(){const ui=useUI();return (<><section className="bottom-rail">
    <div className="legend" aria-label="Matrix colors"><span><i className="w"></i>Weights</span><span><i className="k"></i>Keys</span><span><i className="v"></i>Values</span><span><i className="q"></i>Queries</span><span><i className="a"></i>Attention</span><span><i className="o"></i>Input/output</span><span><i className="omitted"></i>Hidden</span></div>
    <ScaleNote/>
    <button id="export" className="export-button" disabled={!ui.hasTrace||ui.busy} onClick={()=>actions.exportTrace()}>Export trace ↗</button>
  </section></>);}
function StatusBar(){const ui=useUI();return (<><footer><span id="status" role="status">{ui.status}</span><span id="metrics">{ui.metrics}</span></footer></>);}
function ModelDialog(){const ui=useUI();return (<><dialog id="model-dialog" aria-labelledby="model-title">
    <form method="dialog"><button className="dialog-close icon-button" aria-label="Close model picker">×</button></form>
    <h1 id="model-title">Choose a model</h1>
    <label htmlFor="model-choice" className="field-label">Model</label>
    <select id="model-choice" disabled={ui.modelLoading} onChange={event=>{uiStore.set({customModel:event.target.value==="custom"});actions.resetValidation();if(event.target.value==="custom")requestAnimationFrame(()=>document.getElementById("model-url").focus());}}>
      <option value="recommended">SmolLM2 135M · 145 MB (recommended)</option>
      <option value="smollm360" data-url="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf">SmolLM2 360M · 386 MB</option>
      <option value="tinyllama" data-url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf">TinyLlama 1.1B · 669 MB</option>
      <option value="llama1b" data-url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf">Llama 3.2 1B · 808 MB</option>
      <option value="custom">Use a GGUF link…</option>
    </select>
    <div id="custom-model" hidden={!ui.customModel}>
      <label htmlFor="model-url" className="field-label">Hugging Face GGUF link</label>
      <input id="model-url" disabled={ui.modelLoading} onInput={()=>actions.resetValidation()} type="url" spellCheck="false" placeholder="https://huggingface.co/…/model.gguf" />
    </div>
    <p className="model-note">Downloads once. Runs on your computer.</p>
    <div id="validation" className="validation error" role="status" hidden={!ui.validation}><strong>{ui.validation}</strong></div>
    <div id="load-progress" hidden={!ui.progressVisible}><div className="progress-track"><div id="progress-fill" style={{width:(ui.progress*100)+"%"}}></div></div><p id="progress-message">{ui.progressMessage}</p></div>
    <div className="dialog-actions"><button id="check-model" className="primary" disabled={ui.modelLoading} onClick={()=>actions.loadModel()}>{ui.loadLabel}</button></div>
  </dialog></>);}
const AboutDialog=memo(function AboutDialog(){return (<><dialog id="about-dialog" aria-labelledby="about-title">
    <form method="dialog"><button className="dialog-close icon-button" aria-label="Close about">×</button></form>
    <span className="eyebrow">KEY–VALUE CACHE INSPECTION TOOL</span><h1 id="about-title">Meet KeVin.</h1>
    {demo&&<p>Web demo: all displayed data comes from a saved real SmolLM2 135M inference session. Generate reveals up to five recorded tokens; it does not run a model in your browser. Run locally for live inference.</p>}
    <p>Pronounced “Kevin.” An inference-only explorer of a real pretrained model’s attention and KV cache, inspired by littleLLAMA.</p>
    <h3>Read the picture</h3><p>The prompt is processed together in <strong>prefill</strong>. Each <strong>decode</strong> column processes one generated token. Previous keys and values are reused; only the new entries are appended.</p>
    <h3>Every color comes from inference</h3><p>Keys and queries are captured after rotary position encoding. Cache heatmaps use the actual F16 cache operands. Attention colors are probabilities from the model’s softmax. Flash Attention is disabled for inspection.</p>
    <h3>A proxy for a large matrix</h3><p>Each block retains actual cells from its four corners. Corner depths follow the square roots of the real row and column counts. Original indices label the retained cells; shaded bands with ellipses stand for omitted ranges. The proxy grows wider or taller with the matrix, with a bounded prefill proxy followed by all generated entries. No cells are averaged.</p>
    <h3>Fading toward the hidden center</h3><p>Retained cells gradually fade as they approach an omitted range. A shaded gap has no displayed numerical value. Click a cell to inspect its original coordinates and captured precision. <strong>Numbers</strong> controls labels without changing the proxy, and labels appear when zoomed in enough to read.</p>
    <h3>The same attention layout</h3><p>Kᵀ sits beside A, and V beside VA. Caches appear below the divider. New cache entries use the same simple matrix outline. Enable <strong>Math</strong>, then click projection, A, or VA cells to trace their math. With Math unchecked, clicks only focus matrices. Blue orthogonal connectors follow littleLLAMA’s style: solid lines connect displayed operands, dashed lines connect omitted ranges. <strong>Compact cache</strong> uses one shared Kᵀ/V operand and cache snapshot; each query still attends only to its own visible prefix.</p>
    <p>X and Q stay visible. <strong>Weights and K/V</strong> expands the real selected-head Wq/Wk/Wv weights and K/V projections. Orange weight blocks are shared across steps and use the same corner shading. Q/K apply RoPE, and K/V are rounded by the F16 cache write. These calculations run in the model whether the bracket is open or closed. Prefill proxy ranges stay fixed; every generated token is appended visibly. Non-compact decode columns show only added K/V rows and columns. Click a matrix to zoom smoothly.</p>
    <p>Generation uses greedy decoding. Query heads can share a KV head; switching heads preserves the inference timeline. The cache indicator reports logical tensor bytes, excluding allocation padding.</p>
  </dialog></>);});
const RendererPanels=memo(function RendererPanels(){return (<>

  <button id="memory-toggle" aria-controls="memory-panel" aria-expanded="false" hidden>‹ KV memory</button>
  <aside id="memory-panel" className="memory-panel" aria-label="KV tensor memory" hidden>
    <h3>KV tensor memory</h3><div id="memory-context"></div>
    <div id="memory-total"></div><div id="memory-arrays"></div>
  </aside>
  <div id="memory-resize" role="separator" aria-label="Resize memory panel" aria-orientation="vertical" tabIndex="0" hidden></div>
  <svg id="memory-links" aria-hidden="true"></svg>
  <button id="clear-focus" title="Clear selection (Esc)" aria-keyshortcuts="Escape" hidden>Clear selection</button>
  <aside id="inspector" className="inspector" hidden>
    <button id="close-inspector" className="icon-button" aria-label="Close matrix explanation">×</button>
    <span className="eyebrow">INSIDE ATTENTION</span><h2 id="inspect-title"></h2><p id="inspect-description"></p><div id="inspect-shape"></div><p id="inspect-value" hidden></p><div id="inspect-math"></div><button id="focus-matrix">Focus matrix ↗</button>
  </aside>
  <div id="cell-math-overlay" />
  <Toast/>

  <nav id="camera-toolbar" className="camera-toolbar" aria-label="Diagram navigation">
    <button id="zoom-in" aria-label="Zoom in" title="Zoom in (+)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M12 5v14"/></svg></button>
    <button id="zoom-out" aria-label="Zoom out" title="Zoom out (−)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14"/></svg></button>
    <span className="toolbar-separator" aria-hidden="true"></span>
    <button id="recenter" aria-label="Center diagram" title="Center diagram"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="5"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/></svg></button>
    <button id="top" aria-label="Go to input" title="Top · input"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4h14M12 20V8m-5 5 5-5 5 5"/></svg></button>
    <button id="bottom" aria-label="Go to attention output" title="Bottom · attention output"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 20h14M12 4v12m-5-5 5 5 5-5"/></svg></button>
    <button id="overview" aria-label="Fit diagram" title="Overview (0)"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6"/></svg></button>
    <output id="zoom" className="sr-only" aria-label="Zoom level">60%</output>
  </nav>

  </>);});
function App(){
  useLayoutEffect(()=>{startController();},[]);
  return <><Header/><Controls/><Viewport/><RendererPanels/><BottomRail/><StatusBar/><ModelDialog/><AboutDialog/><LocalOnlyDialog/></>;
}
function LocalOnlyDialog(){const ui=useUI(),ref=useRef(null);useLayoutEffect(()=>{if(!ref.current)return;if(ui.localOnlyOpen&&!ref.current.open)ref.current.showModal();if(!ui.localOnlyOpen&&ref.current.open)ref.current.close();},[ui.localOnlyOpen]);if(!demo)return null;return <dialog ref={ref} onClose={()=>uiStore.set({localOnlyOpen:false})} id="local-only-dialog" aria-labelledby="local-only-title"><form method="dialog"><button className="dialog-close icon-button" aria-label="Close local version information">×</button></form><span className="eyebrow">LOCAL ONLY</span><h1 id="local-only-title">Run KeVin locally.</h1><p id="local-only-description">{ui.localOnlyMessage}</p><p>The web demo explores a real SmolLM2 135M recording with a fixed prompt and five generated tokens. Your local version supports custom prompts, other models, and further generation.</p><a className="run-local-link primary-link" href="run-locally.html">Run locally →</a></dialog>;}
window.KevinUI=uiStore;
if(demo)document.body.classList.add('web-demo');
createRoot(document.getElementById('root')).render(<App/>);

function EmptyState(){const ui=useUI();return <div id="empty" className="empty-state" hidden={ui.hasTrace}><h1 id="empty-title">{ui.emptyTitle}</h1><p id="empty-description">{ui.emptyDescription}</p><button id="empty-action" className="primary" hidden={ui.emptyActionHidden} onClick={()=>actions.openModel()}>Choose model</button></div>;}

function Toast(){const ui=useUI();return <div id="toast" className="toast" role="status" hidden={!ui.toast}>{ui.toast}</div>;}


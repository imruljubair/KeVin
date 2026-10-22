/* Loaded only by the static build. The local application has no demo limits. */
(() => {
  'use strict';
  const cache = new Map();
  let manifestPromise, manifest, currentStep = 0, layer = 0, head = 0;
  const ready = () => manifestPromise ||= fetch('data/manifest.json').then(async response => {
    if (!response.ok) throw Error('Could not load the recorded demo. Please reload the page.');
    manifest = await response.json();
    return manifest;
  });
  async function view() {
    const m = await ready(), key = `${layer}:${head}`;
    let captured = cache.get(key);
    if (!captured) {
      const response = await fetch(`data/layer-${layer}-head-${head}.json.gz`);
      if (!response.ok) throw Error('Could not load this recorded layer/head. Please retry.');
      if (!('DecompressionStream' in window)) throw Error('This demo needs a browser with gzip stream support. Try an updated browser or run KeVin locally.');
      captured = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
      if (cache.size >= 4) cache.delete(cache.keys().next().value);
      cache.set(key, captured);
    }
    return {...captured, ...m.frames[currentStep], steps: captured.steps.slice(0, currentStep + 1),
      layer, head, kv_head: captured.kv_head, prompt: m.prompt,
      source: m.source, format: 'kavin-trace-v1', captured_at: m.capturedAt,
      recording: {mode: 'recorded-inference', step: currentStep, limit: m.generationLimit}};
  }
  function localOnly(reason = 'This feature is available when running KeVin locally.') {
    window.KevinUI.set({localOnlyMessage:reason,localOnlyOpen:true});
  }
  function atLimit() { return Boolean(manifest && currentStep >= manifest.generationLimit); }
  async function request(path, command) {
    const m = await ready();
    if (path === '/api/status') return {phase: 'ready', engine_ready: true, model: m.model, message: 'Recorded inference · fixed SmolLM2 demo'};
    if (path === '/api/trace') return view();
    if (path !== '/api/session') throw Error('Run KeVin locally to load models or run live inference.');
    if (command.op === 'prefill') {
      if (command.prompt !== m.prompt) { localOnly('Run KeVin locally to enter your own prefill text.'); return view(); }
      currentStep = 0;
    } else if (command.op === 'step') {
      if (atLimit()) localOnly(`You’ve explored all ${m.generationLimit} recorded tokens. Run KeVin locally to generate more.`);
      else currentStep++;
    } else if (command.op === 'view') {
      const nextLayer = Number(command.layer), nextHead = Number(command.head);
      if (!Number.isInteger(nextLayer) || nextLayer < 0 || nextLayer >= m.model.layers ||
          !Number.isInteger(nextHead) || nextHead < 0 || nextHead >= m.model.heads) throw Error('Invalid recorded layer/head.');
      layer = nextLayer; head = nextHead;
    } else throw Error('Unsupported demo operation.');
    return view();
  }
  function afterRender() {
    window.KevinUI.set({status:`Recorded inference · ${currentStep}/${manifest.generationLimit} generated tokens`,
      metrics:window.KevinUI.getSnapshot().metrics+' · RECORDED'});
  }
  window.KevinDemo = {request, localOnly, atLimit, afterRender};
})();

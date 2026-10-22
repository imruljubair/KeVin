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
    document.querySelector('#local-only-description').textContent = reason;
    const dialog = document.querySelector('#local-only-dialog');
    if (!dialog.open) dialog.showModal();
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
  function refreshControls(state) {
    const limit = atLimit();
    const prompt = document.querySelector('#prompt');
    prompt.readOnly = true;
    prompt.setAttribute('aria-readonly', 'true');
    for (const id of ['step', 'play']) {
      const control = document.querySelector('#' + id);
      control.classList.toggle('local-only', limit);
      control.setAttribute('aria-description', limit ? 'Local only. Activate for installation instructions.' : 'Recorded inference.');
      if (limit) {
        control.disabled = state.busy;
        control.title = 'Local only · run locally to generate more tokens';
      } else control.title = id === 'step' ? 'Reveal the next recorded token' : 'Play recorded tokens';
    }
    document.querySelector('#step').innerHTML = limit ? 'Generate more · local only <span>→</span>' : 'Generate next token <span>→</span>';
    document.querySelector('#prefill').title = 'Replay the fixed recorded prefill from the beginning';
  }
  function afterRender() {
    document.querySelector('#status').textContent = `Recorded inference · ${currentStep}/${manifest.generationLimit} generated tokens`;
    document.querySelector('#metrics').textContent += ' · RECORDED';
  }
  function initUI() {
    document.body.classList.add('web-demo');
    const badge = document.querySelector('.local-badge');
    badge.textContent = 'RECORDED DEMO';
    const guide = document.createElement('a');
    guide.className = 'run-local-link'; guide.href = 'run-locally.html'; guide.textContent = 'Run locally ↗';
    document.querySelector('#about-button').before(guide);
    const dialog = document.createElement('dialog');
    dialog.id = 'local-only-dialog'; dialog.setAttribute('aria-labelledby', 'local-only-title');
    dialog.innerHTML = '<form method="dialog"><button class="dialog-close icon-button" aria-label="Close local version information">×</button></form><span class="eyebrow">LOCAL ONLY</span><h1 id="local-only-title">Run KeVin locally.</h1><p id="local-only-description"></p><p>The web demo explores a real SmolLM2 135M recording with a fixed prompt and five generated tokens. Your local version supports custom prompts, other models, and further generation.</p><a class="run-local-link primary-link" href="run-locally.html">Run locally →</a>';
    document.body.append(dialog);
    const model = document.querySelector('#model-button');
    model.classList.add('local-only'); model.title = 'Local only · load another model'; model.setAttribute('aria-description', 'Local only. Activate for installation instructions.');
    const input = document.querySelector('#prompt');
    input.readOnly = true; input.classList.add('local-only'); input.title = 'Fixed demo prompt · edit prompts in the local version';
    input.addEventListener('click', () => localOnly('Run KeVin locally to enter your own prefill text.'));
    input.addEventListener('keydown', event => {
      if (event.key.length === 1 && !event.ctrlKey && !event.metaKey || ['Backspace', 'Delete'].includes(event.key)) {
        event.preventDefault(); localOnly('Run KeVin locally to enter your own prefill text.');
      }
    });
    const promptLabel = document.querySelector('label[for="prompt"]'); promptLabel.textContent = 'Prompt · local only';
    document.querySelector('#prefill').innerHTML = 'Replay prefill <span>↵</span>';
    document.querySelector('#scale-note').textContent = 'Recorded inference · fixed prompt · 5 generated tokens';
    const about = document.createElement('p');
    about.textContent = 'Web demo: all displayed data comes from a saved real SmolLM2 135M inference session. Generate reveals up to five recorded tokens; it does not run a model in your browser. Run locally for live inference.';
    document.querySelector('#about-title').after(about);
  }
  window.KevinDemo = {request, localOnly, atLimit, refreshControls, afterRender, initUI};
})();

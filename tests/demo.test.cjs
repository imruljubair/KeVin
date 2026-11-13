const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');

function demo() {
  const nodes = {
    '#local-only-description': {textContent: ''},
    '#local-only-dialog': {open: false, showModal() { this.open = true; }},
  };
  const requests = [];
  const context = {window: {DecompressionStream, KevinUI: {
      set(patch){if(patch.localOnlyMessage)nodes['#local-only-description'].textContent=patch.localOnlyMessage;if(patch.localOnlyOpen)nodes['#local-only-dialog'].showModal();},
    }}, Response, DecompressionStream,
    document: {querySelector: selector => nodes[selector]},
    fetch: async url => {
      requests.push(url);
      assert.ok(url.startsWith('data/'), 'Static demo must never request an API');
      const file = path.join(root, 'demo', url);
      return fs.existsSync(file) ? new Response(fs.readFileSync(file)) : new Response('', {status: 404});
    }};
  vm.runInNewContext(fs.readFileSync(path.join(root, 'web/demo.js'), 'utf8'), context);
  return {adapter: context.window.KevinDemo, nodes, requests};
}

for (const preset of ['short','medium','long']) test(`${preset} replay preserves cache and stops at ten decode steps`, async () => {
  const {adapter, nodes, requests} = demo();
  const status = await adapter.request('/api/status');
  assert.equal(status.model.name, 'SmolLM2-135M');
  const choice = status.demoPresets.find(option => option.id === preset);
  let trace = await adapter.request('/api/session', {op:'prefill', preset});
  const initial = trace.tokens.length;
  assert.equal(initial, choice.tokenCount);
  assert.equal(trace.prompt, choice.prompt);
  assert.equal(trace.recording.preset, preset);
  assert.equal(trace.steps.length, 1);
  const firstKey = trace.steps[0].matrices.kc[0][0];
  for (let step = 1; step <= 10; step++) {
    trace = await adapter.request('/api/session', {op: 'step'});
    assert.equal(trace.tokens.length, initial + step);
    assert.equal(trace.steps.length, step + 1);
    assert.equal(trace.steps[0].matrices.kc[0][0], firstKey);
    assert.equal(trace.steps.at(-1).matrices.kc[0].length, trace.tokens.length);
    assert.equal(trace.recording.step, step);
    assert.equal(trace.logical_cache_bytes, trace.tokens.length * 30 * 3 * 64 * 2 * 2);
  }
  assert.equal(adapter.atLimit(), true);
  const capped = await adapter.request('/api/session', {op: 'step'});
  assert.equal(capped.tokens.length, initial + 10);
  assert.equal(nodes['#local-only-dialog'].open, true);
  assert.match(nodes['#local-only-description'].textContent, /locally/);
  assert.equal(requests.filter(url => url.endsWith('.gz')).length, 1, 'Reuse the current captured view');
});

test('layer and head switching preserves replay position, reset restores prefill', async () => {
  const {adapter, nodes} = demo();
  await adapter.request('/api/session', {op: 'step'});
  const trace = await adapter.request('/api/session', {op: 'view', layer: 29, head: 8});
  assert.equal(trace.layer, 29); assert.equal(trace.head, 8); assert.equal(trace.kv_head, 2);
  assert.equal(trace.steps.length, 2); assert.equal(trace.recording.step, 1);
  const unchanged = await adapter.request('/api/session', {op: 'prefill', prompt: 'Custom text'});
  assert.equal(unchanged.recording.step, 1);
  assert.equal(unchanged.prompt, 'Once upon a time');
  assert.equal(nodes['#local-only-dialog'].open, true);
  const reset = await adapter.request('/api/session', {op: 'prefill', prompt: 'Once upon a time'});
  assert.equal(reset.steps.length, 1); assert.equal(reset.layer, 29); assert.equal(reset.head, 8);
  assert.equal(adapter.atLimit(), false);
  await assert.rejects(adapter.request('/api/session', {op: 'view', layer: 30, head: 0}), /Invalid/);
  await assert.rejects(adapter.request('/api/load', {url: 'other-model'}), /locally/);
});

test('switching presets resets generation, retains the inspected head, and isolates cached views', async () => {
  const {adapter, requests} = demo();
  const short = await adapter.request('/api/session', {op:'view', layer:29, head:8});
  await adapter.request('/api/session', {op:'step'});
  const long = await adapter.request('/api/session', {op:'prefill', preset:'long'});
  assert.equal(long.recording.step, 0);
  assert.equal(long.layer, 29);
  assert.equal(long.head, 8);
  assert.ok(long.tokens.length > short.tokens.length);
  assert.equal(long.steps[0].matrices.kc[0].length, long.tokens.length);
  const back = await adapter.request('/api/session', {op:'prefill', preset:'short'});
  assert.equal(back.recording.step, 0);
  assert.equal(back.tokens.length, short.tokens.length);
  assert.deepEqual(requests.filter(url=>url.endsWith('.gz')), [
    'data/short/layer-29-head-8.json.gz', 'data/long/layer-29-head-8.json.gz',
  ]);
});

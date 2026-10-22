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
  const context = {window: {DecompressionStream}, Response, DecompressionStream,
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

test('recorded generation has correct prefix, cache bytes and a hard limit', async () => {
  const {adapter, nodes, requests} = demo();
  const status = await adapter.request('/api/status');
  assert.equal(status.model.name, 'SmolLM2-135M');
  let trace = await adapter.request('/api/trace');
  const initial = trace.tokens.length;
  assert.equal(trace.steps.length, 1);
  const firstKey = trace.steps[0].matrices.kc[0][0];
  for (let step = 1; step <= 5; step++) {
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
  assert.equal(capped.tokens.length, initial + 5);
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

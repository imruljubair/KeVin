# GitHub Pages demo

The browser edition uses a saved real inference session, not browser inference.
It fixes SmolLM2 135M Q8_0, prompt `Once upon a time`, and five generated tokens.
All 30 layers and 9 query heads are recorded. View files are fetched on demand;
visitors do not download the model or the entire recording to open the demo.

Model selection and prompt editing are visibly local-only. Clicking them, or
attempting generation beyond the limit, opens an explanation linking to
`run-locally.html`. Locked actions remain focusable so keyboard users can reach
the same information. Replay prefill restarts the recorded session. Zoom, pan,
optional Math mode, matrix inspection, memory, compact mode, numbers, layer/head selection, and
trace export work in the browser. Math starts unchecked and memory starts
collapsed. Enable Math for paired RoPE selections, symbolic omitted ranges,
and matrix-level formulas. Exported traces include recording metadata.

## Build and preview

```sh
npm ci
npm run build
python3 scripts/build_pages.py
python3 -m http.server 8793 --directory _site
```

Open `http://localhost:8793`. The build also works under a GitHub Pages project
path such as `/KeVin/`. It uses relative asset/data links and has no backend API
dependency. A current browser with `DecompressionStream` support is required
for the gzip recordings. The app reports an actionable error if unsupported.

The local `python3 -m kevin` app does not load `demo.js` and has no demo limits.

## Refresh the recording

After building the real engine and downloading the supported model:

```sh
python3 scripts/record_demo.py --model models/SmolLM2-135M-Q8_0.gguf
python3 scripts/build_pages.py
npm test
```

The recorder uses a separate engine process, so it does not change a running
local session. `demo/data/manifest.json` records model identity, source,
capture time, token-prefix metadata and the generation limit. Each compressed
view contains the actual intermediate matrices for all six replay frames.
The recordings also include pre-RoPE inputs and runtime frequency metadata
for the math cards. The build verifies all expected layer/head files before
publishing and copies the current appearance configuration and SVG branding.

## Publish with GitHub Desktop

The repository's `origin` is `https://github.com/imruljubair/KeVin.git`.
Open this existing repository in GitHub Desktop, commit changes to `main`, and
click **Push origin**. The Pages workflow builds `_site` from the current web
sources and saved data, then deploys it. Model GGUF files, build products, logs,
and the generated `_site` folder are excluded from Git.

Deployment URL: `https://imruljubair.github.io/KeVin/`.

## Preparing a commit

Run the checks before committing:

```sh
npm run build
npm test
python3 -m unittest discover -s tests -p 'test_*.py' -q
python3 scripts/build_pages.py
```

Include frontend sources, the regenerated `web/app.js`, web assets and appearance
settings, tests, documentation, and updated recorded views. The recordings
remain actual captured inference; their added RoPE explanation fields support
the same math in the browser edition. `_site` is a generated preview and stays
ignored; GitHub Actions builds and publishes it after **Push origin**.

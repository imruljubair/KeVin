# React + D3 frontend

The migration preserves KeVin's CSS, fonts, element IDs, matrix proxy geometry,
native inference API, diagram interactions, and recorded-demo limits.

- `frontend/main.jsx`: React header, inference controls, model picker, empty
  state, status/toast messages, and local-only dialog.
- `frontend/ui-store.mjs`: immutable snapshots consumed with
  `useSyncExternalStore`; camera frames do not publish React state updates.
- `frontend/controller.mjs`: inference coordination and the existing
  Canvas/SVG matrix, memory, annotation, selection, and pointer renderer.
- `frontend/visualization.mjs`: D3 camera transforms, legacy-compatible color
  scales, cubic easing, SVG construction, and scene updates.
- `web/demo.js`: recorded-data adapter. React renders its controls/messages.
- `web/app.js`: production bundle, committed so local startup needs no npm
  installation. Edit `frontend/`, then rebuild; do not edit the generated bundle.

React owns application chrome and its state. The diagram and memory panel are
memoized hosts whose descendants and geometry belong to the imperative renderer.
React does not reconcile matrix cells during pan/zoom. SVG annotations use D3
selections; Canvas remains responsible for heatmap pixels. Existing wheel,
pointer, keyboard, and animation timing conventions are retained.

## Run normally

```sh
python3 -m kevin
```

The Python server, native engine, fonts, and compiled frontend remain local.

## Change the frontend

Use Node.js 22 and npm:

```sh
npm ci
npm run build
npm test
```

Refresh the browser after rebuilding. GitHub Pages runs these same build and
test commands, then `python3 scripts/build_pages.py`. Both the local and web
editions use the same React bundle.

Tests cover the existing layout/math/recording behavior and equivalence of D3
heatmap colors, pointer anchoring, and camera easing with the previous renderer.

## Restore the previous frontend

The complete pre-migration frontend and integration files are saved in
`backups/before-react-d3/`; `snapshot.json` identifies the original Git commit.

```sh
python3 scripts/restore_frontend.py --dry-run
python3 scripts/restore_frontend.py
```

The helper saves the active frontend to a fresh recovery folder before restoring
the original web files, Pages builder, and workflow. It prints a command to
reapply that saved frontend. Reload the browser afterward. It does not change
the model, Python package, inference engine, recordings, or font preference.

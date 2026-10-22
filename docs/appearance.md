# Reversible handwriting style

KeVin defaults to Architects Daughter for matrix values (13px), diagram labels,
branding, and explanatory headings. Main headings use 24px; inspector headings
use 20px. Matrix geometry, controls, equations, coordinates and tiny memory
labels retain their original sizing and fonts.

## Internal undo / redo

In `web/index.html`, set the HTML element's `data-font-style` attribute:

- `architects-daughter`: use the handwritten style (current setting).
- `original`: restore the original fonts.

Refresh the page after changing this setting. There is no font selector in the
UI and no browser preference overrides this internal setting.

The original `web/style.css` rules remain intact; the new rules are appended
and apply only with `data-font-style="architects-daughter"`. The font is served locally for offline portability, using a single
explicitly allowed static route under the existing security policy. Its source and OFL license are saved
under `web/fonts/`.

Exact pre-change copies of `web/index.html`, `web/app.js`, and `web/style.css`
plus `kevin/server.py` (originally `kavin/server.py`) are in `docs/style-backups/before-architects-daughter/`. Prefer the internal attribute setting for
visual undo/redo. Restoring those files also discards subsequent edits to them.

An already running server needs a restart to enable the new static font route.

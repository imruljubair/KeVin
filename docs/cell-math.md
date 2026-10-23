# Cell and matrix math

Enable **Math** beside Numbers to inspect calculations. It starts unchecked.
Without Math, clicks focus matrices at 85%; the KV view, numbers, memory panel,
and numeric hover values remain available. Turning Math off clears cards,
ribbons, operand labels, and selection fading.

## Cell selections

Click a cell at detail zoom to show an equation-only graph-paper card. Formulas
use the bundled handwriting font, normally 15px, controlled by
`--formula-text-size` in `web/appearance.css`. Scalars use plain letters; vector
labels and formulas use single bars, such as `|b|`. Small arrows point toward
operands. Upward arrows sit above their text. Indices start at zero.

The card follows the camera, prefers the right side, and moves below the selected
outputs when that side has insufficient space. Above is a fallback at the
viewport's bottom edge. A tall card is limited to the available space. Wheel
and trackpad gestures over cards or variable labels pan the diagram;
Ctrl/Command+wheel zooms. Escape, Clear selection, or clicking the selected
cell again dismisses the explanation.

Final numeric outputs and evaluated intermediate numbers are omitted. Known
model settings are substituted into frequency and position formulas; unavailable
settings stay symbolic. Powers use real superscripts. Native MathML renders
all formulas without a CDN or external math service.

- **K/Q:** selecting either RoPE partner highlights both outputs as `a` and `b`.
  The paired weight rows are `|c|` and `|d|`; the input column is `|e|`.
  Two dot products produce `u` and `v`, then a full 2 × 2 sine/cosine matrix
  produces the output pair. Position and frequency calculations remain visible.
  Adjacent and split-half pairing use the actual row partners. Bias, head
  normalization, cache rounding, and extended scaling are represented when present.
- **V:** weight-row/input-column dot product.
- **Scores:** post-RoPE key/query dot product divided by the square root of
  the head dimension.
- **A:** exponent of the selected score divided by the sum of exponentials
  of causally visible scores; future entries are masked.
- **VA:** value-row/attention-column dot product.
- **Caches:** projection copies or transposed cached keys.

Omitted ranges also show symbolic cards without inventing a representative
value. RoPE keeps separate paired row sets and avoids `first()`/`second()` in
its standard adjacent and split-half explanations. Paired ribbons use separate
landings; if both members share an omitted band, each lands on its own half.
The two-slot omission bands give paired labels space. Matrix names fade during
cell selections, and ribbons pass over matrix names.

## Matrix names

With Math enabled, clicking a matrix name shows its matrix-level formula in
the same paper style. Examples include `V = Wv · X`,
`Scores = Kᵀ · Q / sqrt(head dimension)`, and `VA = V · A`.
K/Q include their RoPE stage. Weight-name selections show the projection that
uses those weights. Without Math, names focus the matrix instead.

## Captured data and implementation

The engine captures the actual inputs and frequency settings of its RoPE nodes
in each step's `explanation.q` and `explanation.k`. Stored K values come from
the actual F16 cache; formulas never reconstruct a dot product from incomplete,
rounded corner proxies. After updating an existing local engine, rebuild with
`python3 scripts/setup.py`, restart, and prefill again to capture these fields.
Older traces use symbolic parameter names when captures are unavailable.

All 270 recorded web views include matching pre-RoPE captures. Their original
matrix values and recording metadata were preserved. The Pages build ships the
same UI, formulas, appearance settings, and logo as the local app.

`frontend/cell-math.mjs` describes formulas and operand ranges;
`frontend/controller.mjs` anchors cards, labels, highlights and ribbons to the
camera; `web/style.css` supplies the graph-paper styling.

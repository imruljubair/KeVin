# Appearance experiments

Edit **`web/appearance.css`**. The commented `:root` section at the top is the
configuration: font families, independent text sizes, matrix/cell/connection
borders, panel borders and corner radii. The CSS below that section connects
those values to the existing UI.

1. Start KeVin with `python3 -m kevin` from the repository folder.
2. Edit a value, save the file, and refresh `http://127.0.0.1:8765/`.
3. No npm build is needed for appearance edits, including Canvas cell borders.
   When upgrading an older running server, restart it once to load the new
   stylesheet route. Further appearance edits need only a page refresh.

For example:

```css
--cell-text-size: 14px;          /* Values inside the matrices. */
--matrix-title-size: 15px;       /* K, Q, V, Scores, A, etc. */
--matrix-border-width: 2px;      /* Outer contours. */
--cell-font: ui-monospace, monospace;
```

Keep `px` on sizes and widths; use nonnegative numbers. The renderer reads its
Canvas widths at startup. Diagram contours compensate for camera zoom. Matrix
cell geometry is unchanged, so oversized text can crowd the cells. Focused
numbers are fitted using their configured rendered font size.

Architects Daughter is bundled locally, so the default works offline on other
machines. System fonts use the listed fallbacks. A new named font needs to be
installed on each machine or bundled before it can be portable.

The default configuration keeps Architects Daughter. Appearance settings now
take precedence over the older `data-font-style` attribute. To experiment with
system text, change the font variables here; there is no font selector in the UI.

The initial settings are saved in `docs/style-backups/appearance-defaults.css`.
To reset, copy that file over `web/appearance.css` and refresh. Save additional
copies of your own experiments whenever desired. The complete implementation from before
this configuration exists in Git history. Older font and pre-React snapshots
remain in `docs/style-backups/before-architects-daughter/` and
`backups/before-react-d3/`; restoring them also discards subsequent UI changes.

The Pages build copies the same appearance file, so published builds use the
same settings. Local experiments appear publicly only after publishing them.

The current defaults use 15px formulas, a 260px expanded memory panel, 12px
memory circles, and larger memory indices. Math and the memory panel start off
and collapsed respectively. These view defaults live in the frontend code;
`--memory-panel-width` and `--memory-dot-size` control their expanded appearance.
The graph-paper background and label-arrow layout live in `web/style.css`.

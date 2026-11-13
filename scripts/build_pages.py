#!/usr/bin/env python3
"""Build the static GitHub Pages edition without Python or model dependencies."""
import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify_recordings(data):
    catalog = json.loads((data / 'manifest.json').read_text())
    if (catalog['schema'] != 'kevin-recorded-presets-v1'
            or [item['id'] for item in catalog['presets']] != ['short', 'medium', 'long']
            or catalog['defaultPreset'] != 'short'):
        raise ValueError('Invalid demo preset catalog')
    model = None
    for preset in catalog['presets']:
        folder = data / preset['id']
        manifest = json.loads((folder / 'manifest.json').read_text())
        frames = manifest['frames']
        if (manifest['schema'] != 'kevin-recorded-demo-v1'
                or manifest['generationLimit'] != 10 or len(frames) != 11
                or manifest['prompt'] != preset['prompt']
                or any(len(frame['tokens']) != preset['tokenCount'] + step for step, frame in enumerate(frames))):
            raise ValueError(f"Invalid recording for {preset['id']}")
        if model is not None and manifest['model'] != model:
            raise ValueError('All demo presets must use the same model')
        model = manifest['model']
        for layer in range(model['layers']):
            for head in range(model['heads']):
                path = folder / f'layer-{layer}-head-{head}.json.gz'
                view = json.loads(gzip.decompress(path.read_bytes()))
                if (view['layer'] != layer or view['head'] != head or view['tokens'] != frames[-1]['tokens']
                        or len(view['steps']) != len(frames)):
                    raise ValueError(f'Incorrect recorded view: {path}')


def build(output):
    output = output.resolve()
    # Avoid deleting a caller's unrelated files or accepting a source directory.
    if output == ROOT or ROOT / 'web' == output or output in ROOT.parents or output.exists() and any(output.iterdir()) and not (output / '.kevin-pages-build').exists():
        raise ValueError('Choose an empty output folder or a previous KeVin Pages build')
    verify_recordings(ROOT / 'demo/data')
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / '.kevin-pages-build').touch()
    for name in ['style.css', 'appearance.css', 'app.js', 'matrix_proxy.js', 'favicon.svg', 'logo.svg', 'demo.js', 'demo.css', 'run-locally.html', 'guide.css', 'vendor-licenses.txt']:
        shutil.copy2(ROOT / 'web' / name, output / name)
    shutil.copytree(ROOT / 'web/fonts', output / 'fonts')
    shutil.copytree(ROOT / 'demo/data', output / 'data')
    html = (ROOT / 'web/index.html').read_text()
    html = re.sub(r'(href|src)="/([^\"]*)"', r'\1="\2"', html)
    html = html.replace('</head>', '<link rel="stylesheet" href="demo.css">\n</head>')
    html = html.replace('<script src="app.js" defer>', '<script src="demo.js" defer></script><script src="app.js" defer>')
    html = html.replace('Watch a real language model\'s keys, values, and attention evolve, one token at a time.', 'Explore a recorded SmolLM2 inference session. Run KeVin locally for your own prompts and models.')
    (output / 'index.html').write_text(html)
    css = (output / 'style.css').read_text().replace('url("/fonts/', 'url("fonts/')
    (output / 'style.css').write_text(css)
    (output / '.nojekyll').touch()
    print(f'Built static KeVin demo: {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '_site')
    build(parser.parse_args().output)

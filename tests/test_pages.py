"""Exercise the actual Pages package, including project-relative assets."""
import re
import tempfile
import unittest
from pathlib import Path

from scripts.build_pages import build


class PagesPackage(unittest.TestCase):
    def test_published_pages_resolve_assets_under_a_project_path(self):
        with tempfile.TemporaryDirectory() as folder:
            output = build(Path(folder) / "KeVin")
            for name in ("appearance.css", "logo.svg", "favicon.svg",
                         "fonts/ArchitectsDaughter-Regular.ttf", ".nojekyll"):
                self.assertTrue((output / name).is_file(), name)
            for page in ("index.html", "run-locally.html"):
                html = (output / page).read_text()
                for target in re.findall(r'(?:src|href)="([^"]+)"', html):
                    if target.startswith(("https:", "http:", "#")):
                        continue
                    self.assertFalse(target.startswith("/"), target)
                    self.assertTrue((output / target).exists(), target)
            self.assertEqual(len(list((output / "data").glob("layer-*-head-*.json.gz"))), 270)
            self.assertIn("demo.js", (output / "index.html").read_text())

    def test_build_preserves_an_unrelated_nonempty_output_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            sentinel = output / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(ValueError):
                build(output)
            self.assertEqual(sentinel.read_text(), "keep")

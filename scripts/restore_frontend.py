#!/usr/bin/env python3
"""Restore a frontend snapshot, preserving the active version for undo/redo."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ['web', 'scripts/build_pages.py', '.github/workflows/pages.yml']


def copy_snapshot(source, destination):
    for name in FILES:
        src, dst = source / name, destination / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'backups/before-react-d3')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = args.snapshot.resolve()
    for name in FILES:
        if not (source / name).exists():
            raise SystemExit(f'Incomplete snapshot: missing {name}')
    if not (source / 'web/index.html').is_file() or not (source / 'web/app.js').is_file():
        raise SystemExit('Snapshot must include the web entry point and application script')
    if args.dry_run:
        print(f'Would restore {source}: {", ".join(FILES)}; active files will be backed up first.')
        return
    recovery = ROOT / 'backups' / datetime.now(timezone.utc).strftime('frontend-before-restore-%Y%m%d-%H%M%S-%f')
    recovery.mkdir(parents=True, exist_ok=False)
    copy_snapshot(ROOT, recovery)
    (recovery / 'snapshot.json').write_text(json.dumps({'restored_from': str(source)}, indent=2))
    copy_snapshot(source, ROOT)
    print('Frontend restored. Reload KeVin in your browser.')
    print(f'To undo this restoration: python3 scripts/restore_frontend.py --snapshot "{recovery}"')


if __name__ == '__main__':
    main()

# Before React + D3

This is the exact frontend and integration snapshot captured before migration.
It includes `web/`, the Pages build script/workflow, and the Python server source.
`snapshot.json` records the original Git commit.

To restore the frontend while first saving the currently active implementation:

```sh
python3 scripts/restore_frontend.py
```

Run that from the repository root. The Python backend is deliberately not
overwritten by the restore helper; it was not changed by this migration.
See `docs/react-d3.md` for development and undo/redo instructions.

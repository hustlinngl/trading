# Windows EXE packaging

The canonical Windows build is produced by GitHub Actions with PyInstaller from the current release tree.

## Output

The artifact contains:

- `SakuraSignalTerminal.exe` — the read-only local dashboard.
- `config.yaml` — runtime configuration, kept external so it can be edited safely.
- `SETUP_WINDOWS.md` and `README.md` — usage/reference.
- `RUN.txt` — quick launch notes.
- `models/assets/<SYMBOL>/` — validated signal bundles used by the packaged terminal.

The EXE opens the browser automatically and binds the local terminal to localhost by default.

## Model bundles

Validated model and deployment bundles are shipped beside the executable. The build trains and validates the configured 15 core assets, then the packaged smoke test verifies that all 15 bundles are discoverable by the frozen application. The models remain external to the single-file binary so provenance and compatibility checks remain inspectable.

## Rebuild

Push a tag such as `v0.9.18` or manually start the **Build Windows EXE** workflow from GitHub Actions.


Build target: Windows x64 / Python 3.11.

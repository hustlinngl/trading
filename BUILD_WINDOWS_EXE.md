# Windows EXE packaging

The canonical Windows build is produced by GitHub Actions with PyInstaller from the frozen 0.9.35 release tree.

## Output

The artifact contains:

- `SakuraSignalTerminal.exe` — the read-only local dashboard.
- `config.yaml` — runtime configuration, kept external so it can be edited safely.
- `SETUP_WINDOWS.md` and `README.md` — usage/reference.
- `RUN.txt` — quick launch notes.

The EXE opens the browser automatically and binds the local terminal to localhost by default.

## Model bundles

Model and deployment bundles are not embedded in the executable. They stay external so the deployed artifact can use the repository's provenance/compatibility checks without rebuilding the EXE for every model generation.

## Rebuild

Push a tag such as `v0.9.35` or manually start the **Build Windows EXE** workflow from GitHub Actions.


Build target: Windows x64 / Python 3.11.

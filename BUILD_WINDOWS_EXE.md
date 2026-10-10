# Windows EXE packaging

The canonical Windows x64 package is produced by GitHub Actions with PyInstaller. Do not distribute an EXE based only on a successful PyInstaller step: the workflow must also pass the frozen-process smoke test and packaged dashboard/HTTP contract test.

## Package contents

The portable artifact includes `SakuraSignalTerminal.exe`, `config.yaml`, usage documentation, `MODEL_READINESS.txt`, the bundled closed-candle cache and per-symbol model bundles under `models/assets/<SYMBOL>/`.

The executable binds the local dashboard to localhost and is read-only. Configuration and model bundles stay beside the executable so provenance and compatibility gates remain inspectable.

## Model readiness is separate from software readiness

The build trains and checks the 15 configured core spot assets. A model is deployable only when its temporal holdout, economic, specialist and provenance checks pass. If software packaging succeeds but no model passes those gates, the app is explicitly **research-only** and must expose zero public LONG/SHORT signals. Do not lower gates just to create signals.

## Build and verify

Push a release tag such as `v0.9.45` or manually dispatch the **Build Windows EXE** workflow in GitHub Actions. Verify that the run is for the intended commit, reaches `Upload Windows package`, and has a downloadable package artifact. Inspect `MODEL_READINESS.txt` before describing the package as model-ready.

Target runtime: Windows x64, Python 3.11. Source launch and test instructions are in `SETUP_WINDOWS.md`.

# Final cleanup handoff — V6 / 0.9.15

The repository now contains the hardened research core plus autonomous orchestration, data utilities, memory/state components, UI controller, strict live/paper inference, regression tests, benchmark controls and CI automation.

## Status
- Engineering focus: research validity, economic realism, provenance, promotion safety and operational correctness.
- Trading edge: unproven; synthetic benchmark is negative and must not be treated as production evidence.
- Default operating mode: paper + sandbox.
- Public-data research path: Binance Vision for long history, Kraken for bounded cross-exchange validation.
- Final holdout is intended to remain untouched by tuning.

## Secrets
Never commit real API keys, tokens, private keys or credentials. Use .env locally or GitHub Actions Secrets. .env.example intentionally contains empty placeholders.

## Final AI cleanup
Reconcile any remaining duplicate/legacy modules against the canonical V6 cognition path, run the full test suite and compile/import checks, verify the GitHub Actions workflow, and remove any redundant compatibility files only after checking import references.

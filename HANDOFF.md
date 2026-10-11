# HANDOFF — per la prossima IA

Obiettivo del proprietario: prodotto finito, operabile **solo dal terminale** (addestramento incluso), senza vedere/modificare codice. Lavoro interamente delegato all'IA, compatibile col piano Claude free.

## Come si opera (stato attuale)
- `python -m ai_trading_lab.cockpit` — menu interattivo (stato, doctor, dati, pipeline, scan, esiti, scorecard, benchmark, test, dashboard).
- Sottocomandi: `status`, `doctor`, `scan`, `outcomes`, `scorecard`, `test`, `dashboard`, `pipeline [--live-bars N] [--holdout-frac F]`.
- Dopo `pip install -e .` anche il comando `cockpit`.
- CI/automazione: `ci.yml`, `build-windows-exe.yml` (training 15 asset + EXE), `intensive-real-research.yml`, `signal-journal.yml` (ogni 6h), `autonomous-research.yml`.

## Invarianti (NON violare)
1. Mai allentare i gate di readiness per produrre segnali (DSR>=0.80, PBO<=0.20, stabilità, placebo, accordo policy+specialista 3–24h).
2. Nessun endpoint di ordini; PAPER_ONLY/SANDBOX di default.
3. Solo candele chiuse come input dei modelli; ticker realtime solo display.
4. Non committare `data/`, `models/`, `logs/`; nessun segreto nel codice.
5. Modifica minima e isolata + test di regressione; allineare VERSION/pyproject/`__init__`/README/CHANGELOG se cambia la release.
6. Una build riuscita non implica un edge profittevole.

## Fatto in questa sessione
- `src/ai_trading_lab/cockpit.py` + `tests/test_cockpit.py` + entry `cockpit` in pyproject.

## Da fare (priorità consigliata)
1. Eseguire `pytest -q` completo in CI e correggere eventuali regressioni (pytest non era installabile nella sandbox della sessione: il test cockpit è stato verificato solo via smoke test).
2. Workflow `workflow_dispatch` che esegue `cockpit pipeline` e persiste il ledger sul branch `research-ledger`.
3. Comando `cockpit report`: diagnosi automatica dei gate falliti per asset in linguaggio chiaro (cosa manca per diventare PRONTO).
4. Verificare le chiavi reali di `deployment_manifest.json` (`ready`, `checks`/motivi) e adattare `_failed()` in cockpit.py.
5. Installer one-shot (`start_ui.*`) che lanci il cockpit.
6. Documentare il cockpit in README/START_HERE.

## Convenzioni commit
Trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` e `Claude-Session: <url sessione>`.

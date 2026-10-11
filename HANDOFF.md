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

## Fatto
- `cockpit.py` + `tests/test_cockpit.py` + entry `cockpit` in pyproject.
- Suite completa verificata: 259 passed (prima delle modifiche di questa sessione).
- Chiavi reali del manifest verificate (`ready`, `checks` dict di bool, inclusi i `live_policy_*`): `_failed()` compatibile.
- `cockpit report`: diagnosi in chiaro dei gate falliti per asset (`EXPLAIN`, con test che ne verifica la copertura).
- Workflow `cockpit-pipeline.yml` (workflow_dispatch): ripristina il ledger, esegue `cockpit pipeline`, persiste il ledger su `research-ledger` anche in caso di fallimento.
- Documentazione cockpit in README e START_HERE.

## Da fare
1. Eseguire una volta `Cockpit Pipeline` su Actions e verificare il ledger persistito (non testabile dalla sessione: rete/exchange).
2. Installer one-shot (`start_ui.*`) che lanci il cockpit.
3. Se si aggiungono check a `deployment.py`, aggiungere la spiegazione in `cockpit.EXPLAIN` (il test lo impone per i `live_*`).

## Convenzioni commit
Trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` e `Claude-Session: <url sessione>`.

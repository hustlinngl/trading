# Prompt da incollare alla prossima IA (Claude, account free)

Copia tutto il blocco qui sotto in una nuova sessione collegata al repo `hustlinngl/trading`.

```
Lavori sul repo hustlinngl/trading (Adaptive AI Trading Lab / Sakura Signal Terminal, release 0.9.46).
Obiettivo del proprietario: prodotto finito, operabile SOLO da terminale (training incluso) senza vedere o modificare codice; lavoro interamente tuo, decidi tu cosa ha valore, senza schemi preimposti. Rispondi in italiano, in modo conciso.

1. Leggi per primo HANDOFF.md (stato, invarianti, TODO), poi docs/PROJECT_MAP.md. Non leggere altro se non serve: risparmia usage.
2. Parti dal branch claude/terminal-cockpit (contiene src/ai_trading_lab/cockpit.py, tests/test_cockpit.py, HANDOFF.md). Se non è ancora in main, apri/mergia la PR.
3. Esegui in ordine i TODO di HANDOFF.md:
   a) lancia la suite test (CI o pytest) e correggi regressioni;
   b) verifica le chiavi reali di deployment_manifest.json e adatta cockpit._failed();
   c) workflow workflow_dispatch che esegue `cockpit pipeline` e persiste il ledger sul branch research-ledger;
   d) comando `cockpit report` con diagnosi in chiaro dei gate falliti per asset;
   e) documenta il cockpit in README/START_HERE.
4. Invarianti: mai allentare i gate di readiness; nessun endpoint di ordini; PAPER_ONLY/SANDBOX; non committare data/models/logs; nessun segreto; modifiche piccole + test; allinea VERSION/pyproject/__init__/README/CHANGELOG se cambi release.
5. Commit su branch dedicato, trailer: "Co-Authored-By: <tuo modello> <noreply@anthropic.com>" e "Claude-Session: <url>". Push dopo ogni passo completato.
6. Delega il lavoro pesante (training, ricerca) a GitHub Actions, non alla sessione.
7. Quando hai poco usage: aggiorna HANDOFF.md (fatto / da fare), fai push e termina.
```

"""Cockpit: console unica per operare l'intero ciclo senza toccare codice.

Uso:
    python -m ai_trading_lab.cockpit            # menu interattivo
    python -m ai_trading_lab.cockpit status     # stato modelli/evidenze
    python -m ai_trading_lab.cockpit pipeline   # dati -> training -> stato
    python -m ai_trading_lab.cockpit report     # perché ogni asset non è PRONTO
    python -m ai_trading_lab.cockpit --help

Sola lettura sul mercato: nessun ordine viene mai inviato.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
FAILED_KEYS = ("failed_checks", "blockers", "reasons")
MAIN = [sys.executable, "-m", "ai_trading_lab.main"]


# Spiegazioni in chiaro dei controlli di readiness (chiavi reali di deployment.py).
# I gate NON vengono modificati: qui si spiega soltanto cosa manca.
EXPLAIN = {
    "required_engine_artifacts_present": "Mancano file del modello base: rilancia il training dell'asset.",
    "model_provenance_present": "Il modello non registra l'impronta dei dati: va riaddestrato con la pipeline attuale.",
    "deployment_semantics_match": "Le regole di esecuzione sono cambiate dopo il training: riaddestra.",
    "holdout_report_present": "Manca il report di holdout del modello base: completa il training.",
    "holdout_provenance_match": "Il report di holdout non corrisponde ai dati/modello attuali: riaddestra.",
    "holdout_trade_support": "Nell'holdout il modello base ha fatto troppo pochi trade per essere valutabile.",
    "positive_holdout_return": "Il rendimento netto sull'holdout è negativo: nessun edge dimostrato.",
    "profit_factor": "Profit factor dell'holdout sotto la soglia minima.",
    "holdout_drawdown": "Drawdown dell'holdout peggiore del limite consentito.",
    "risk_adjusted_utility": "Utilità corretta per il rischio sotto la soglia minima.",
    "duration_artifact_present": "Manca lo specialista 3-24h: rilancia il training dell'asset.",
    "duration_report_present": "Manca il report di training dello specialista 3-24h.",
    "duration_provenance_match": "Lo specialista 3-24h non è coerente con modello base/dati: riaddestra.",
    "duration_production_ready": "Lo specialista 3-24h non ha superato i propri criteri di holdout.",
    "live_policy_holdout_report_present": "Manca il report holdout della policy live (gate + specialista insieme).",
    "live_policy_schema_match": "Il report holdout della policy live ha uno schema non riconosciuto.",
    "live_policy_gate_applied": "Il report non è stato prodotto con il gate live rigoroso.",
    "live_specialist_gate_applied": "Il report non include il filtro dello specialista 3-24h.",
    "live_policy_provenance_match": "Il report della policy live non corrisponde ai dati/modello attuali.",
    "live_policy_holdout_pristine": "L'holdout è stato consultato più volte: evidenza non più incontaminata.",
    "live_policy_trade_support": "La policy live ha prodotto troppo pochi trade nell'holdout.",
    "live_policy_positive_net_return": "La policy live ha rendimento netto non positivo nell'holdout.",
    "live_policy_profit_factor": "Profit factor della policy live sotto soglia.",
    "live_policy_drawdown": "Drawdown della policy live oltre il limite.",
    "live_policy_utility": "Utilità corretta per il rischio della policy live sotto soglia.",
    "live_policy_dsr": "Deflated Sharpe Ratio sotto 0.80: il risultato può essere dovuto al caso.",
    "live_policy_pbo": "Probabilità di overfitting (PBO) sopra 0.20.",
    "live_policy_stability": "Risultati instabili tra i sotto-periodi.",
    "live_policy_negative_control": "Il controllo placebo non è stato battuto: l'edge non è distinguibile dal rumore.",
    "live_policy_fold_support": "Troppo pochi fold di validazione per un giudizio affidabile.",
}


def _run(args: list[str]) -> int:
    print("\n$ " + " ".join(args), flush=True)
    return subprocess.call(args)


def _read(path: Path) -> dict:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def _failed(manifest: dict) -> list[str]:
    """Nomi dei controlli falliti, tollerante a formati diversi del manifest."""
    out: list[str] = []
    checks = manifest.get("checks")
    if isinstance(checks, dict):
        out += [str(k) for k, v in checks.items() if not v]
    for key in FAILED_KEYS:
        v = manifest.get(key)
        if isinstance(v, (list, tuple)):
            out += [str(x) for x in v]
    return list(dict.fromkeys(out))


def status() -> int:
    assets = ROOT / "models" / "assets"
    print("=== STATO MODELLI ===")
    if not assets.is_dir():
        print("Nessun modello addestrato. Esegui: pipeline")
    else:
        ready = 0
        for d in sorted(p for p in assets.iterdir() if p.is_dir()):
            m = _read(d / "deployment_manifest.json")
            ok = bool(m.get("ready"))
            ready += ok
            failed = _failed(m)
            line = f"{'PRONTO    ' if ok else 'NON PRONTO'}  {d.name}"
            if failed:
                line += "  <- falliti: " + ", ".join(failed[:4]) + ("…" if len(failed) > 4 else "")
            print(line)
        print(f"\nPronti per segnali: {ready}")
        if ready == 0:
            print("Modalità research-only: nessun segnale LONG/SHORT verrà pubblicato (comportamento corretto).")
    ledger = _read(ROOT / "data" / "research_ledger.json")
    if ledger:
        print(f"\nLedger ricerca: {ledger.get('total_trials', 0)} trial, "
              f"{len(ledger.get('holdout_accesses', []))} accessi holdout")
    j = ROOT / "logs" / "live_signal_history.jsonl"
    if j.exists():
        with j.open(encoding="utf-8", errors="replace") as fh:
            print(f"Journal segnali: {sum(1 for _ in fh)} righe")
    return 0


def report() -> int:
    """Diagnosi per asset: quali gate falliscono e cosa significa, in chiaro."""
    assets = ROOT / "models" / "assets"
    print("=== REPORT READINESS ===")
    if not assets.is_dir():
        print("Nessun modello addestrato. Esegui: pipeline")
        return 0
    blocked = 0
    for d in sorted(p for p in assets.iterdir() if p.is_dir()):
        m = _read(d / "deployment_manifest.json")
        if not m:
            print(f"\n{d.name}: manifest assente o illeggibile -> esegui il training dell'asset.")
            blocked += 1
            continue
        if m.get("ready"):
            print(f"\n{d.name}: PRONTO (tutti i gate superati; non implica profitto futuro).")
            continue
        blocked += 1
        failed = _failed(m)
        print(f"\n{d.name}: NON PRONTO - {len(failed)} gate falliti")
        for name in failed:
            print(f"  - {name}: {EXPLAIN.get(name, 'controllo non documentato nel cockpit.')}")
        if not failed:
            print("  - nessun dettaglio nel manifest: riesegui il training.")
    print(f"\nAsset non pronti: {blocked}. I gate non vengono mai allentati: "
          "se mancano evidenze, il sistema resta in WAIT/research-only.")
    return 0


def pipeline(live_bars: int, holdout: float) -> int:
    steps = [
        MAIN + ["doctor"],
        MAIN + ["bootstrap-live-data", "--live-bars", str(live_bars), "--workers", "4"],
        MAIN + ["train-complete-all", "--holdout-frac", str(holdout)],
    ]
    for s in steps:
        rc = _run(s)
        if rc != 0:
            print(f"\nStep fallito (codice {rc}). Interrotto.")
            return rc
    return status()


ACTIONS = {
    "1": ("Stato modelli e evidenze", lambda: status()),
    "r": ("Report: perché gli asset non sono PRONTI", lambda: report()),
    "2": ("Diagnostica ambiente (doctor)", lambda: _run(MAIN + ["doctor"])),
    "3": ("Scarica dati live (bootstrap)", lambda: _run(MAIN + ["bootstrap-live-data", "--live-bars", "1200", "--workers", "4"])),
    "4": ("Pipeline completa: dati + training + stato", lambda: pipeline(1200, 0.15)),
    "5": ("Scan segnali (Top 5)", lambda: _run(MAIN + ["live-scan"])),
    "6": ("Risolvi esiti segnali maturi", lambda: _run(MAIN + ["live-outcomes"])),
    "7": ("Scorecard statistica del journal", lambda: _run(MAIN + ["scorecard", "--min-resolved", "30"])),
    "8": ("Benchmark controlli", lambda: _run(MAIN + ["benchmark", "--benchmark-bars", "1200"])),
    "9": ("Test suite", lambda: _run([sys.executable, "-m", "pytest", "-q"])),
    "d": ("Avvia dashboard (sola lettura)", lambda: _run([sys.executable, "signal_dashboard.py"])),
}


def menu() -> int:
    while True:
        print("\n=== ADAPTIVE AI COCKPIT ===")
        for k, (label, _) in ACTIONS.items():
            print(f" {k}) {label}")
        print(" q) Esci")
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return 0
        if choice in {"q", "quit", "exit"}:
            return 0
        action = ACTIONS.get(choice)
        if action:
            action[1]()
        else:
            print("Scelta non valida.")


def main() -> int:
    p = argparse.ArgumentParser(description="Console operativa (nessun ordine, sola lettura)")
    sub = p.add_subparsers(dest="cmd")
    for name in ("status", "report", "doctor", "scan", "outcomes", "scorecard", "test", "dashboard"):
        sub.add_parser(name)
    pl = sub.add_parser("pipeline")
    pl.add_argument("--live-bars", type=int, default=1200)
    pl.add_argument("--holdout-frac", type=float, default=0.15)
    a = p.parse_args()
    table = {
        "status": status,
        "report": report,
        "doctor": lambda: _run(MAIN + ["doctor"]),
        "scan": lambda: _run(MAIN + ["live-scan"]),
        "outcomes": lambda: _run(MAIN + ["live-outcomes"]),
        "scorecard": lambda: _run(MAIN + ["scorecard"]),
        "test": lambda: _run([sys.executable, "-m", "pytest", "-q"]),
        "dashboard": lambda: _run([sys.executable, "signal_dashboard.py"]),
    }
    if a.cmd is None:
        return menu()
    if a.cmd == "pipeline":
        return pipeline(a.live_bars, a.holdout_frac)
    return int(table[a.cmd]() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

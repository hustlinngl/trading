"""Cockpit: console unica per operare l'intero ciclo senza toccare codice.

Uso:
    python -m ai_trading_lab.cockpit            # menu interattivo
    python -m ai_trading_lab.cockpit status     # stato modelli/evidenze
    python -m ai_trading_lab.cockpit pipeline   # dati -> training -> stato
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
    for name in ("status", "doctor", "scan", "outcomes", "scorecard", "test", "dashboard"):
        sub.add_parser(name)
    pl = sub.add_parser("pipeline")
    pl.add_argument("--live-bars", type=int, default=1200)
    pl.add_argument("--holdout-frac", type=float, default=0.15)
    a = p.parse_args()
    table = {
        "status": status,
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

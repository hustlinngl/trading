"""Sakura Atlas — minimal native SPY research studio.

A fixed Sakura palette, one experiment action, saved history. No web server,
API keys, broker connection, order entry or automatic publication of signals.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
# A frozen one-file EXE unpacks beside a temporary directory; never write
# model evidence there or depend on the current working directory.
STORAGE = (Path.home() / "SakuraAtlas") if getattr(sys, "frozen", False) else PROJECT
sys.path.insert(0, str(PROJECT / "src"))

from ai_trading_lab.spy_research import analyze_spy, list_studies  # noqa: E402
from ai_trading_lab.spy_shadow import track_spy  # noqa: E402

try:
    import tkinter as tk
except ImportError:
    tk = None

BG = "#100D17"
SURFACE = "#191322"
SURFACE_ALT = "#22192C"
LINE = "#34243E"
PINK = "#FF83BA"
PINK_PALE = "#FFD3E7"
TEXT = "#F4EAF5"
MUTED = "#B6A3C0"
DANGER = "#FF9C9C"


def as_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.2f}%"


def summarize(report: dict) -> str:
    test = report.get("test", {})
    val = report.get("validation", {})
    evidence = report.get("test_evidence", {})
    dates = report.get("split", {})
    return "\n".join([
        "SPY / RESEARCH STUDY",
        "",
        f"Period                   {dates.get('test_start', '?')} to {dates.get('test_end', '?')}",
        "Daily bars               Adjusted for dividends and splits",
        "Decision / execution     Close signal, next-open entry, following-open exit",
        "",
        "INDEPENDENT-EVIDENCE CHECK / RESEARCH ONLY",
        f"Assessment               {evidence.get('verdict', 'NOT EVALUATED')}",
        evidence.get("explanation", "No evidence gate has been run for this historic study."),
        "",
        "OUT-OF-SAMPLE CHECK (exploratory, not pristine)",
        f"Strategy, after costs    {as_percent(test.get('compounded_net_return'))}",
        f"Buy & hold, after costs  {as_percent(report.get('test_buy_and_hold_net_return'))}",
        f"Difference               {as_percent(report.get('test_excess_vs_buy_and_hold'))}",
        f"Maximum drawdown         {as_percent(test.get('max_drawdown'))}",
        f"Executed paper trades    {test.get('trades', 0)} / {test.get('sessions', 0)} sessions",
        f"Trade win rate           {as_percent(test.get('win_rate_after_cost'))}",
        f"Edge confidence interval {str(evidence.get('mean_daily_excess_90pct_block_bootstrap', 'not available'))}",
        f"Beats timing placebo     {evidence.get('beats_random_timing_placebo', 'not evaluated')}",
        "",
        "EARLIER VALIDATION PERIOD",
        f"Net return               {as_percent(val.get('compounded_net_return'))}",
        f"Paper trades             {val.get('trades', 0)}",
        "",
        "STATE: RESEARCH ONLY",
        "No model is authorized to produce live trading recommendations or orders.",
        "This one experiment cannot demonstrate a lasting market advantage.",
        "",
        f"Saved experiment         {report.get('run_id', '-')}",
    ])


class AtlasApp(tk.Tk if tk is not None else object):
    def __init__(self):
        if tk is None:
            raise RuntimeError("This Python installation lacks tkinter (Tk desktop support)")
        super().__init__()
        self.title("Sakura Atlas")
        self.geometry("1070x700")
        self.minsize(900, 610)
        self.configure(bg=BG)
        self.events = queue.Queue()
        self.busy = False
        self.studies = []
        self._build()
        self._refresh_studies()
        self.after(100, self._drain_queue)

    def _label(self, parent, text, *, size=11, color=TEXT, bold=False, **kwargs):
        return tk.Label(
            parent, text=text, bg=kwargs.pop("bg", BG), fg=color,
            font=("Segoe UI", size, "bold" if bold else "normal"),
            anchor=kwargs.pop("anchor", "w"), justify="left", **kwargs,
        )

    def _build(self):
        header = tk.Frame(self, bg=BG, padx=42, pady=27)
        header.pack(fill="x")
        self._label(header, "SAKURA  /  ATLAS", size=12, color=PINK, bold=True).pack(anchor="w")
        self._label(header, "Market research, beautifully focused.", size=25, color=TEXT, bold=True).pack(anchor="w", pady=(10, 4))
        self._label(
            header, "One instrument. One honest experiment. Your research stays on this computer.",
            size=11, color=MUTED,
        ).pack(anchor="w")

        tk.Frame(self, bg=LINE, height=1).pack(fill="x", padx=42)
        content = tk.Frame(self, bg=BG, padx=42, pady=28)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        left = tk.Frame(content, bg=SURFACE, padx=28, pady=26)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        self._label(left, "01 / EXPLORE", bg=SURFACE, color=PINK, bold=True).pack(anchor="w")
        self._label(left, "SPY · S&P 500 ETF", bg=SURFACE, size=21, bold=True).pack(anchor="w", pady=(12, 6))
        self._label(
            left,
            "Adjusted daily history. Chronological training.\n"
            "Paper-only economics including estimated trading costs.",
            bg=SURFACE, size=11, color=MUTED,
        ).pack(anchor="w")
        self.run_button = tk.Button(
            left, text="Analyze SPY  ↗", command=self._start,
            bg=PINK, fg=BG, activebackground=PINK_PALE, activeforeground=BG,
            relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI", 13, "bold"), padx=25, pady=14,
        )
        self.run_button.pack(anchor="w", pady=(28, 12))
        self.follow_button = tk.Button(
            left, text="Check paper observations", command=self._follow,
            bg=SURFACE_ALT, fg=PINK_PALE,
            activebackground=LINE, activeforeground=TEXT,
            relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI", 11, "bold"), padx=22, pady=11,
        )
        self.follow_button.pack(anchor="w", pady=(0, 14))
        self.status_var = tk.StringVar(value="Ready to analyze.")
        self.status_label = self._label(
            left, "", bg=SURFACE, color=MUTED, size=11, textvariable=self.status_var,
            wraplength=450,
        )
        self.status_label.pack(anchor="w", pady=(0, 20))

        tk.Frame(left, bg=LINE, height=1).pack(fill="x", pady=(0, 18))
        self._label(left, "HOW THE RESULT IS JUDGED", bg=SURFACE, color=PINK_PALE, size=11, bold=True).pack(anchor="w")
        for explanation in (
            "Train on the oldest 60% of available sessions.",
            "Validate on the following 20%; evaluate on the latest 20%.",
            "Compare after-cost returns against dividend-aware buy & hold.",
            "Store a new, dated study and model. Never overwrite old runs.",
        ):
            self._label(
                left, "•  " + explanation, bg=SURFACE, color=MUTED, size=10,
                wraplength=460, pady=5,
            ).pack(anchor="w")

        right = tk.Frame(content, bg=SURFACE_ALT, padx=24, pady=24)
        right.grid(row=0, column=1, sticky="nsew")
        self._label(right, "02 / STUDY ARCHIVE", bg=SURFACE_ALT, color=PINK, bold=True).pack(anchor="w")
        self._label(right, "Every run, preserved.", bg=SURFACE_ALT, size=17, bold=True).pack(anchor="w", pady=(10, 14))
        self.archive = tk.Listbox(
            right, height=5, bg=BG, fg=TEXT, selectbackground="#653851",
            selectforeground=TEXT, borderwidth=0, highlightthickness=1,
            highlightbackground=LINE, font=("Consolas", 10), activestyle="none",
            exportselection=False,
        )
        self.archive.pack(fill="x")
        self.archive.bind("<<ListboxSelect>>", self._selection_changed)
        self._label(right, "REPORT", bg=SURFACE_ALT, color=PINK_PALE, bold=True, size=10).pack(anchor="w", pady=(20, 10))
        report_frame = tk.Frame(right, bg=BG)
        report_frame.pack(fill="both", expand=True)
        self.details = tk.Text(
            report_frame, bg=BG, fg=TEXT, insertbackground=PINK,
            relief="flat", bd=0, highlightthickness=0,
            wrap="word", font=("Consolas", 9), padx=14, pady=14,
            state="disabled",
        )
        self.details.pack(fill="both", expand=True)
        footer = tk.Frame(self, bg=BG, padx=42, pady=14)
        footer.pack(fill="x")
        self._label(
            footer, "RESEARCH ONLY   •   No broker integration or automatic trading.   •   No profitability guarantee.",
            color=MUTED, size=9,
        ).pack(anchor="w")

    def _show(self, content: str):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", content)
        self.details.configure(state="disabled")

    def _refresh_studies(self, latest_id: str | None = None):
        self.studies = list_studies(STORAGE)
        self.archive.delete(0, "end")
        for row in self.studies:
            test = row.get("test", {})
            self.archive.insert(
                "end", f"{row.get('run_id', '')[:15]}    {as_percent(test.get('compounded_net_return'))}"
            )
        if self.studies:
            index = next(
                (i for i, row in enumerate(self.studies) if row.get("run_id") == latest_id), 0
            )
            self.archive.selection_set(index)
            self.archive.see(index)
            self._show(summarize(self.studies[index]))
        else:
            self._show("No study has been run yet.\n\nUse Analyze SPY to create your first locally saved research experiment.")

    def _selection_changed(self, _event=None):
        selection = self.archive.curselection()
        if selection and selection[0] < len(self.studies):
            self._show(summarize(self.studies[selection[0]]))

    def _start(self):
        self._launch("study")

    def _follow(self):
        self._launch("shadow")

    def _launch(self, mode: str):
        if self.busy:
            return
        self.busy = True
        self.run_button.configure(state="disabled")
        self.follow_button.configure(state="disabled")
        self.status_var.set(
            "Running the SPY study and saving its evidence..."
            if mode == "study" else
            "Checking completed sessions against the saved research model..."
        )

        def worker():
            try:
                result = analyze_spy(STORAGE) if mode == "study" else track_spy(STORAGE)
                self.events.put((mode, result))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True, name=f"atlas-{mode}").start()

    def _drain_queue(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                self.busy = False
                self.run_button.configure(state="normal")
                self.follow_button.configure(state="normal")
                if kind == "study":
                    self.status_var.set("Study saved. Its evidence appears on the right.")
                    self._refresh_studies(value["run_id"])
                elif kind == "shadow":
                    latest = value.get("latest") or {}
                    self.status_var.set(
                        "Paper observation saved." if value["recorded_new_observation"]
                        else "No new observation; recorded history preserved."
                    )
                    self._show("\\n".join([
                        "SPY / FORWARD PAPER OBSERVATIONS",
                        "",
                        "These are research observations, NEVER market orders.",
                        "Predictions are recorded once after a completed session.",
                        "Missed sessions are not retroactively filled.",
                        "",
                        f"Latest observed day      {latest.get('session', '-')}",
                        f"Paper direction          {latest.get('shadow_side', '-')}",
                        f"Probability up           {latest.get('probability_up', 0):.1%}",
                        f"Sessions tracked         {value['observations']}",
                        f"Resolved paper trades    {value['resolved_shadow_trades']}",
                        "Net paper return         " + as_percent(
                            value.get("net_return_on_resolved_shadow_trades")
                        ),
                        "",
                        "All observations are exploratory. Forward evidence",
                        "does not imply profitability or permission to trade.",
                        "",
                        f"Active research model    {value['run_id']}",
                    ]))
                else:
                    self.status_var.set("The operation failed; earlier research is unchanged.")
                    self._show("STUDY NOT COMPLETED\\n\\n" + value)
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)


def main():
    p = argparse.ArgumentParser(description="Native Sakura Atlas SPY research app (never places orders)")
    p.add_argument("--check", action="store_true", help="Import smoke check without opening a window")
    args = p.parse_args()
    if args.check:
        print("Sakura Atlas import OK (SPY research only, no orders)")
        return 0
    if tk is None:
        raise SystemExit("tkinter is unavailable. Install the Tk desktop support for your Python.")
    AtlasApp().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

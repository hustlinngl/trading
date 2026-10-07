from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_trading_lab.config import load_settings
from ai_trading_lab.live import scan_top5, write_live_snapshot
from ai_trading_lab.main import system_doctor, synthetic_data, test_base_holdout
from ai_trading_lab.benchmarks import evaluate_suite

st.set_page_config(page_title="Adaptive AI Trading Lab", page_icon="🧠", layout="wide")


def _read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return {}


def _state_badge(value: bool, yes: str = "READY", no: str = "WAIT") -> str:
    return yes if value else no


settings = load_settings()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {padding: 1.1rem 1.3rem; border: 1px solid rgba(127,127,127,.22); border-radius: 16px; margin-bottom: 1rem;}
    .muted {opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

doctor = system_doctor(settings, ".")
paper_state = _read_json("logs/paper_last.json")
research_state = _read_json("logs/intensive_real/intensive_real_research_summary.json")
promotion_state = _read_json("models/promotion_state.json")

st.markdown(
    f"""
    <div class="hero">
      <h1 style="margin-bottom:.25rem;">Adaptive AI Trading Lab</h1>
      <div class="muted">Research-first adaptive trading control center · conservative signal generation · no autonomous live execution</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Live Radar", "Research & Evidence", "Diagnostics", "Safety"])

with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Version", doctor["version"])
    c2.metric("Research", _state_badge(doctor.get("research_ready", False)))
    c3.metric("Live data", _state_badge(doctor.get("live_data_ready", False)))
    c4.metric("Paper only", _state_badge(bool(doctor.get("paper_only", True)), "YES", "NO"))
    c5.metric("Sandbox", _state_badge(bool(doctor.get("sandbox", True)), "YES", "NO"))

    st.subheader("System posture")
    posture = pd.DataFrame(
        [
            ["Core", doctor.get("core_ready", False), "Python / numerical stack"],
            ["Research", doctor.get("research_ready", False), "CV, tuning and statistical diagnostics"],
            ["Live data", doctor.get("live_data_ready", False), "Public market-data connectivity"],
            ["UI", doctor.get("ui_live_ready", False), "Dashboard dependencies"],
        ],
        columns=["Subsystem", "Ready", "Meaning"],
    )
    st.dataframe(posture, use_container_width=True, hide_index=True)

    latest_signal = paper_state.get("signal", "FLAT")
    st.subheader("Latest paper assessment")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Signal", latest_signal)
    pc2.metric("Confidence", f"{100*float(paper_state.get('confidence', 0.0)):.1f}%")
    pc3.metric("Expected return", f"{100*float(paper_state.get('expected_return', 0.0)):.3f}%")
    pc4.metric("Data age", f"{float(paper_state.get('data_age_minutes', 0.0)):.1f}m")

    if paper_state:
        reason = paper_state.get("reason") or []
        if reason:
            st.caption(" · ".join(str(x) for x in reason))
    else:
        st.info("No paper assessment has been recorded yet.")

with tabs[1]:
    st.subheader("Live Radar")
    st.caption("Public-data scan with data-quality, lineage, model-consensus, memory and adaptive time-window gates.")
    if st.button("Scan current market", type="primary"):
        with st.spinner("Reading public market data and evaluating strict signal gates…"):
            results = scan_top5(settings, ".")
        write_live_snapshot(results, ".")
        rows = []
        for r in results:
            row = r.to_dict()
            row["confidence_pct"] = 100.0 * float(row.pop("confidence", 0.0))
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows)[
                ["symbol", "status", "signal", "confidence_pct", "expected_return", "price", "reason_codes"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        snapshot = Path("logs/live_snapshot.json")
        if snapshot.exists():
            try:
                rows = json.loads(snapshot.read_text(encoding="utf-8"))
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    frame["confidence_pct"] = 100.0 * frame["confidence"].astype(float)
                    st.dataframe(
                        frame[["symbol", "status", "signal", "confidence_pct", "expected_return", "price", "reason_codes"]],
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:
                st.error(f"Could not read live snapshot: {exc}")
        else:
            st.info("Run a scan to populate the radar.")

with tabs[2]:
    st.subheader("Research & Evidence")
    r1, r2, r3 = st.columns(3)
    r1.metric("Assets completed", int(research_state.get("assets_completed", 0)))
    r2.metric("Assets failed", int(research_state.get("assets_failed", 0)))
    r3.metric("Promotion", "DISABLED")

    if research_state:
        reports = research_state.get("reports", []) or []
        cards = []
        for report in reports:
            evidence = report.get("evidence", {}) if isinstance(report, dict) else {}
            cards.append(
                {
                    "symbol": report.get("symbol", "?"),
                    "rows": report.get("rows", 0),
                    "holdout_return": (report.get("holdout_tuned") or {}).get("total_return", 0.0),
                    "trades": (report.get("holdout_tuned") or {}).get("trades", 0),
                    "verdict": evidence.get("verdict", "UNKNOWN"),
                    "evidence_score": evidence.get("evidence_score", 0.0),
                }
            )
        if cards:
            st.dataframe(pd.DataFrame(cards), use_container_width=True, hide_index=True)
    else:
        st.info("No intensive real-data research summary has been recorded yet.")

    if promotion_state:
        st.caption(
            f"Last recorded champion score: {promotion_state.get('score', 'n/a')} · "
            f"dataset: {promotion_state.get('data_fingerprint', 'n/a')}"
        )

with tabs[3]:
    st.subheader("Diagnostics")
    d1, d2 = st.columns(2)
    with d1:
        st.write("Synthetic holdout")
        if st.button("Run deterministic smoke test"):
            df = synthetic_data(settings, 1200)
            try:
                st.json(test_base_holdout(df, settings, 0.20))
            except Exception as exc:
                st.error(str(exc))
    with d2:
        st.write("Benchmark controls")
        if st.button("Run benchmark suite"):
            try:
                result = evaluate_suite(settings, n=1200)
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.caption("Controls are diagnostics, not evidence of a production trading edge.")
            except Exception as exc:
                st.error(str(exc))

with tabs[4]:
    st.subheader("Safety")
    st.warning(
        "This project is intentionally research-first. Autonomous research can create evidence and proposals, "
        "but does not receive production order authority."
    )
    safety = {
        "paper_only": bool(settings.paper_only),
        "sandbox": bool(settings.sandbox),
        "strict_signal_mode": bool(settings.signal_only_mode),
        "trade_window_required": bool(settings.trade_window_required_for_signal),
        "deployment_manifest_required": bool(settings.require_deployment_manifest_for_signal),
        "short_borrow_bps_per_bar": float(settings.short_borrow_bps_per_bar),
        "max_daily_loss_pct": float(settings.max_daily_loss_pct),
        "risk_per_trade": float(settings.risk_per_trade),
    }
    st.json(safety)

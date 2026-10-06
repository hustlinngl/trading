from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
from ai_trading_lab.config import load_settings
from ai_trading_lab.live import scan_top5, write_live_snapshot
from ai_trading_lab.main import system_doctor, synthetic_data, test_base_holdout

st.set_page_config(page_title="Adaptive AI Trading Lab",page_icon="🧠",layout="wide")
settings=load_settings()
st.title("Adaptive AI Trading Lab")
st.caption("Research-first adaptive trading control center — paper/sandbox by default.")

tabs=st.tabs(["Overview","Data & Tests","Live Radar","Safety"])
with tabs[0]:
    doctor=system_doctor(settings,".")
    c1,c2,c3=st.columns(3)
    c1.metric("Version",doctor["version"]); c2.metric("Core ready","YES" if doctor["core_ready"] else "NO"); c3.metric("Paper only","YES" if doctor["paper_only"] else "NO")
    st.info("Trading edge remains unproven. Use the untouched holdout and cost-stressed research gates before treating any candidate as deployable.")
with tabs[1]:
    st.subheader("Synthetic smoke test")
    if st.button("Run demo holdout"):
        df=synthetic_data(settings,1200)
        try: st.json(test_base_holdout(df,settings,0.20))
        except Exception as exc: st.error(str(exc))
with tabs[2]:
    if st.button("Scan now"):
        with st.spinner("Reading public market data…"):
            results=scan_top5(settings,".")
        write_live_snapshot(results,".")
        st.dataframe([r.to_dict() for r in results],use_container_width=True)
with tabs[3]:
    st.warning("This UI is intentionally restricted to paper/sandbox behavior. Real credentials must never be stored in the repository.")
    st.code(json.dumps({"paper_only":settings.paper_only,"sandbox":settings.sandbox},indent=2))

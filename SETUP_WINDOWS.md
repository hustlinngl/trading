# Adaptive AI Trading Lab V6 — Windows

Install Python 3.11+ 64-bit, then:

```powershell
python -m pip install -e ".[full]"
python -m streamlit run app.py
```

Recommended flow: Data → Train & Test → Live Radar. The project is paper/sandbox by default. The adaptive specialist targets 3–24h, with a soft preference for 3–4h; it is not a hard duration constraint.

Run validation with:

```powershell
python -m pytest -q
python -m ai_trading_lab.main demo
python -m ai_trading_lab.main doctor
```

Never commit API keys. Use environment variables or GitHub Actions Secrets.

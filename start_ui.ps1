Set-Location $PSScriptRoot
python -m pip install -e ".[dashboard]"
python -m streamlit run app.py

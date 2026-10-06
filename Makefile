install:
	python -m pip install -e .[full]

test:
	python -m pytest -q

demo:
	python -m ai_trading_lab.main demo

research:
	python -m ai_trading_lab.main research

discover:
	python -m ai_trading_lab.main discover

optimize:
	python -m ai_trading_lab.main optimize

auto-update:
	python -m ai_trading_lab.main auto-update

grow:
	python -m ai_trading_lab.main grow

autonomous:
	python -m ai_trading_lab.main autonomous

evolve:
	python -m ai_trading_lab.main evolve

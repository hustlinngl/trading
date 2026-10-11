import json

from ai_trading_lab import cockpit


def test_status_without_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cockpit, "ROOT", tmp_path)
    assert cockpit.status() == 0
    assert "Nessun modello" in capsys.readouterr().out


def test_status_reports_ready_and_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cockpit, "ROOT", tmp_path)
    a = tmp_path / "models" / "assets" / "BTC_USDT"
    b = tmp_path / "models" / "assets" / "ETH_USDT"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "deployment_manifest.json").write_text(json.dumps({"ready": True}))
    (b / "deployment_manifest.json").write_text(json.dumps({"ready": False, "checks": {"dsr": False, "pbo": True}}))
    cockpit.status()
    out = capsys.readouterr().out
    assert "PRONTO      BTC_USDT" in out
    assert "NON PRONTO  ETH_USDT" in out and "dsr" in out
    assert "Pronti per segnali: 1" in out


def test_status_corrupt_manifest_is_not_ready(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cockpit, "ROOT", tmp_path)
    d = tmp_path / "models" / "assets" / "X"
    d.mkdir(parents=True)
    (d / "deployment_manifest.json").write_text("{broken")
    cockpit.status()
    assert "NON PRONTO" in capsys.readouterr().out


def test_pipeline_stops_on_first_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(cockpit, "_run", lambda a: calls.append(a) or 3)
    assert cockpit.pipeline(10, 0.1) == 3
    assert len(calls) == 1

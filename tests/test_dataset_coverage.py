from __future__ import annotations

from ai_trading_lab.config import load_settings
from ai_trading_lab.real_universe import DEFAULT_CRYPTO_CORE


def test_default_dataset_coverage_is_broad():
    settings = load_settings("config.yaml")

    assert len(DEFAULT_CRYPTO_CORE) >= 40
    assert len(DEFAULT_CRYPTO_CORE) == len(set(DEFAULT_CRYPTO_CORE))
    assert settings.lookback_bars >= 30_000
    assert settings.live_lookback_bars >= 1_200
    assert settings.broad_crypto_symbol_cap >= 500
    assert settings.intraday_research_top_n >= 30
    assert len(settings.live_symbols) >= 15
    assert len(settings.live_symbols) == len(set(settings.live_symbols))


def test_signal_first_ui_hides_secondary_telemetry():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML

    assert 'class="nav nav-minimal"' in html
    assert 'id="coverageSummary"' in html
    assert 'class="pick-stats pick-stats-compact"' in html
    assert '<div class="k">Expected</div>' in html
    assert '["Edge",pct(d.robust_directional_edge,2)]' not in html
    assert '["Score",num(d.score,2)]' not in html
    assert '.secondary-telemetry{display:none!important}' in html
    assert '<style>\n</style>\n</style>' not in html

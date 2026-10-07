from __future__ import annotations

import pandas as pd

try:
    import optuna
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False

from .engine import AdaptiveEngine
from .features import make_oos_features
from .policy import make_actions
from .evaluation import run_configured_backtest
from .objectives import robust_performance_utility

def optimize_policy(train: pd.DataFrame, validation: pd.DataFrame, settings, trials: int = 30) -> dict:
    """Tune decision gates on a validation set after fitting the model on earlier data.

    This is policy optimization, not model training; callers should keep an untouched final holdout.
    """
    if not HAVE_OPTUNA:
        raise RuntimeError('Optuna is missing. Install the project dependencies.')
    engine = AdaptiveEngine(settings); engine.fit(train)
    feat = make_oos_features(train, validation, settings.horizon_bars, external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1))
    bt_df = validation.copy(); bt_df['atr_14'] = feat['atr_14']
    def objective(trial):
        pt = trial.suggest_float('probability_threshold', 0.52, 0.75)
        edge = trial.suggest_float('min_expected_return', 0.0003, 0.006, log=True)
        decision = trial.suggest_float('decision_threshold', 0.05, 0.45)
        meta = trial.suggest_float('meta_threshold', 0.50, 0.70)
        stop = trial.suggest_float('stop_atr_mult', 1.0, 3.0)
        rr = trial.suggest_float('take_profit_rr', 1.2, 4.0)
        actions = make_actions(engine, feat, settings, pt, edge, decision, meta)
        res = run_configured_backtest(bt_df, actions, settings, stop_atr_mult=stop, take_profit_rr=rr, cost_multiplier=1.0)
        stress = run_configured_backtest(bt_df, actions, settings, stop_atr_mult=stop, take_profit_rr=rr, cost_multiplier=1.5)
        if res.stats['trades'] < 10 or res.stats['max_drawdown'] < -0.25:
            return -10.0 + res.stats['max_drawdown']
        base_utility=robust_performance_utility(res.stats,min_trades=10,max_drawdown=-0.25)
        stress_utility=robust_performance_utility(stress.stats,min_trades=10,max_drawdown=-0.25)
        return float(0.70*base_utility+0.30*stress_utility)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=settings.seed))
    study.optimize(objective, n_trials=int(trials), show_progress_bar=False)
    return {'best_value': float(study.best_value), 'best_params': study.best_params, 'trials': len(study.trials)}

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .efficiency import audit_features
from .execution_semantics import validate_execution_alignment
from .features import make_features
from .labels import triple_barrier_labels
from .memory import AnalogMemory
from .meta import MetaPolicy, side_aware_meta_target
from .models import SignalModel
from .policy import decide_actions
from .regimes import RegimeDetector


@dataclass
class EngineArtifacts:
    features: pd.DataFrame
    labels: pd.DataFrame
    regimes: pd.Series
    model: SignalModel
    short_model: SignalModel
    memory: AnalogMemory
    short_memory: AnalogMemory
    meta: MetaPolicy
    short_meta: MetaPolicy
    predictions: pd.DataFrame


def directional_target_from_returns(target_ret: pd.Series) -> pd.Series:
    """Create a binary success target from executable, realized position returns."""
    returns = pd.to_numeric(target_ret, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return returns.gt(0.0).astype(float).where(returns.notna())


def execution_aligned_targets(
    df: pd.DataFrame,
    horizon_bars: int,
    pt_atr: float,
    sl_atr: float,
    *,
    side: int = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build one side's targets from the same executable barrier/time-stop path.

    Both heads use position PnL as the regression target. The short head has its
    own downside-profit/upside-stop barriers; it does not treat 1 - p_up as a
    probability of success. Ambiguous OHLC collisions stay missing.
    """
    ordered = df.sort_index().copy()
    if ordered.index.has_duplicates:
        raise ValueError("duplicate_timestamps")
    labels = triple_barrier_labels(
        ordered,
        int(horizon_bars),
        float(pt_atr),
        float(sl_atr),
        side=int(side),
    )
    target_returns = pd.to_numeric(
        labels["tb_return"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    direction = directional_target_from_returns(target_returns)
    return labels, target_returns, direction


def _pre_calibration_core_mask(index, calibration_start, purge_bars):
    """Return rows strictly before the purged calibration boundary.

    Timestamp conversion/search errors deliberately propagate. Falling back to
    the full index would silently put calibration observations into meta-training.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return pd.Series(dtype=bool, index=index)
    calibration_timestamp = pd.Timestamp(calibration_start)
    calibration_position = int(index.searchsorted(calibration_timestamp, side="left"))
    core_end = max(0, calibration_position - max(0, int(purge_bars)))
    return pd.Series(np.arange(len(index)) < core_end, index=index, dtype=bool)


def _new_signal_model(settings):
    model = SignalModel(
        settings.seed,
        xgb_estimators=getattr(settings, "xgb_estimators", 240),
        lgbm_estimators=getattr(settings, "lgbm_estimators", 240),
        hist_max_iter=getattr(settings, "hist_max_iter", 260),
    )
    model.conformal_level = float(getattr(settings, "conformal_level", 0.90))
    return model


def _new_memory(settings):
    return AnalogMemory(
        k=getattr(settings, "memory_k", 32),
        exclusion_bars=int(
            getattr(
                settings,
                "memory_exclusion_bars",
                max(1, getattr(settings, "validation_purge_bars", settings.horizon_bars)),
            )
        ),
    )


def _assemble_predictions(
    base,
    short_base,
    regime,
    analog,
    short_analog,
    regime_persistence,
    meta_long,
    meta_short,
    actions,
    scores,
):
    pred = base.copy()
    pred["p_short_raw"] = short_base["p_up_raw"]
    pred["p_short"] = short_base["p_up"]
    pred["short_probability_baseline"] = short_base["probability_baseline"]
    pred["short_expected_return"] = short_base["expected_return"]
    pred["short_expected_return_lcb"] = short_base["expected_return_lcb"]
    pred["short_expected_return_ucb"] = short_base["expected_return_ucb"]
    pred["short_model_disagreement"] = short_base["model_disagreement"]
    pred["short_return_disagreement"] = short_base["return_disagreement"]
    pred["regime"] = regime
    pred["analog_edge"] = analog["edge"]
    pred["analog_agreement"] = analog["agreement"]
    pred["analog_dispersion"] = analog["dispersion"]
    pred["analog_n"] = analog["n"]
    pred["short_analog_edge"] = short_analog["edge"]
    pred["short_analog_agreement"] = short_analog["agreement"]
    pred["short_analog_dispersion"] = short_analog["dispersion"]
    pred["short_analog_n"] = short_analog["n"]
    pred["regime_persistence"] = regime_persistence
    pred["meta_success"] = meta_long
    pred["short_meta_success"] = meta_short
    pred["long_score"] = scores.attrs.get("long_scores", np.full(len(pred), np.nan))
    pred["short_score"] = scores.attrs.get("short_scores", np.full(len(pred), np.nan))
    pred["score"] = scores
    pred["action"] = actions
    return pred.replace([np.inf, -np.inf], np.nan)


class AdaptiveEngine:
    def __init__(self, settings):
        self.settings = settings
        self.regimes = RegimeDetector(
            settings.seed, n_init=getattr(settings, "regime_n_init", 5)
        )
        self.model = _new_signal_model(settings)
        self.short_model = _new_signal_model(settings)
        self.memory = _new_memory(settings)
        self.short_memory = _new_memory(settings)
        self.meta = MetaPolicy(settings.seed)
        self.short_meta = MetaPolicy(settings.seed)
        self.meta_regimes = None

    def features(self, df):
        """Build the canonical feature frame used by research and live inference."""
        features, _, _ = make_features(
            df,
            self.settings.horizon_bars,
            external_feature_lag_bars=getattr(self.settings, "external_feature_lag_bars", 1),
        )
        return features

    def load(self, out_dir):
        """Load a persisted bundle into this engine instance."""
        loaded = type(self).load_bundle(self.settings, out_dir)
        self.__dict__.update(loaded.__dict__)
        return self

    def fit(self, df):
        validate_execution_alignment(self.settings)
        df = df.sort_index().copy()
        features, _, _ = make_features(
            df,
            self.settings.horizon_bars,
            external_feature_lag_bars=getattr(self.settings, "external_feature_lag_bars", 1),
        )
        labels, target_ret, y = execution_aligned_targets(
            df,
            self.settings.horizon_bars,
            self.settings.pt_atr,
            self.settings.sl_atr,
            side=1,
        )
        short_labels, short_target_ret, short_y = execution_aligned_targets(
            df,
            self.settings.horizon_bars,
            self.settings.pt_atr,
            self.settings.sl_atr,
            side=-1,
        )

        cal_n = max(48, int(max(1, y.notna().sum()) * 0.15))
        valid_positions = np.flatnonzero(y.notna().to_numpy())
        selection_mask = pd.Series(False, index=features.index)
        if len(valid_positions) > cal_n + 100:
            selection_mask.iloc[valid_positions[:-cal_n]] = True
        else:
            selection_mask.loc[y.index] = y.notna()

        pruned, efficiency_report = audit_features(
            features,
            y,
            corr_threshold=float(getattr(self.settings, "efficiency_corr_threshold", 0.995)),
            selection_mask=selection_mask,
        )
        self.feature_efficiency = efficiency_report
        purge_bars = int(
            getattr(self.settings, "validation_purge_bars", self.settings.horizon_bars)
        )
        self.model.fit(pruned, y, target_ret, purge_bars=purge_bars)
        self.short_model.fit(
            pruned, short_y, short_target_ret, purge_bars=purge_bars
        )

        model_features = pruned.reindex(index=features.index)
        long_cal_oos = getattr(self.model, "calibration_oos_", None)
        short_cal_oos = getattr(self.short_model, "calibration_oos_", None)
        calibration_starts = [
            frame.index[0]
            for frame in (long_cal_oos, short_cal_oos)
            if frame is not None and not frame.empty
        ]
        core_mask = pd.Series(True, index=features.index, dtype=bool)
        if calibration_starts:
            first_calibration = min(calibration_starts)
            core_mask = _pre_calibration_core_mask(
                features.index, first_calibration, purge_bars
            )

        # Both regime detectors and both meta-training memories are fitted only
        # on the pre-calibration core. The calibration window never determines
        # the unsupervised representation used to score its own rows.
        self.regimes.fit(features.loc[core_mask])
        self.meta_regimes = self.regimes
        regime = self.regimes.transform(features)
        regime_persistence = self.regimes.persistence(features)
        regime_probabilities = self.regimes.semantic_probabilities(features)

        self.memory.fit(
            model_features,
            target_ret,
            information_weighted=bool(
                getattr(self.settings, "memory_information_weighted", False)
            ),
        )
        self.short_memory.fit(
            model_features,
            short_target_ret,
            information_weighted=bool(
                getattr(self.settings, "memory_information_weighted", False)
            ),
        )
        analog = self.memory.query_many(
            model_features,
            exclude_self=True,
            exclusion_bars=int(
                getattr(
                    self.settings,
                    "memory_exclusion_bars",
                    max(1, getattr(self.settings, "validation_purge_bars", self.settings.horizon_bars)),
                )
            ),
        )
        short_analog = self.short_memory.query_many(
            model_features,
            exclude_self=True,
            exclusion_bars=int(
                getattr(
                    self.settings,
                    "memory_exclusion_bars",
                    max(1, getattr(self.settings, "validation_purge_bars", self.settings.horizon_bars)),
                )
            ),
        )

        meta_memory = _new_memory(self.settings)
        short_meta_memory = _new_memory(self.settings)
        if calibration_starts:
            meta_memory.fit(
                model_features.loc[core_mask],
                target_ret.loc[core_mask],
                information_weighted=bool(
                    getattr(self.settings, "memory_information_weighted", False)
                ),
            )
            short_meta_memory.fit(
                model_features.loc[core_mask],
                short_target_ret.loc[core_mask],
                information_weighted=bool(
                    getattr(self.settings, "memory_information_weighted", False)
                ),
            )

        fee_bps = float(getattr(self.settings, "fee_bps", 0.0))
        slippage_bps = float(getattr(self.settings, "slippage_bps", 0.0))
        impact_bps = float(getattr(self.settings, "impact_bps_per_sqrt", 0.0))
        participation = float(np.clip(getattr(self.settings, "max_participation_pct", 0.10), 0.0, 1.0))
        edge_bps = max(0.0, float(getattr(self.settings, "min_edge_after_cost_bps", 5.0)))
        base_cost = (
            2.0 * (fee_bps + slippage_bps)
            + 2.0 * impact_bps * np.sqrt(participation)
            + edge_bps
        ) / 10000.0
        borrow_bps = float(max(0.0, getattr(self.settings, "short_borrow_bps_per_bar", 0.0)))
        holding_bars = int(max(1, getattr(self.settings, "max_holding_bars", 96)))
        short_cost = base_cost + borrow_bps * holding_bars / 10000.0

        def fit_side_meta(calibration_oos, side_returns, meta_model, side_memory, cost_hurdle):
            if calibration_oos is None or calibration_oos.empty:
                return
            idx = calibration_oos.index.intersection(features.index)
            idx = idx.intersection(side_returns.index)
            if len(idx) == 0:
                return
            realized = pd.to_numeric(side_returns.reindex(idx), errors="coerce")
            valid = realized.notna().to_numpy() & np.isfinite(realized.to_numpy(dtype=float))
            if int(valid.sum()) < 100:
                return

            labels_meta = side_aware_meta_target(
                realized.to_numpy(dtype=float),
                np.full(len(idx), float(cost_hurdle)),
            )
            if np.unique(labels_meta[valid]).size < 2:
                return
            side_features = features.loc[idx]
            side_regime = self.regimes.transform(side_features)
            side_persistence = self.regimes.persistence(side_features)
            side_regime_probabilities = self.regimes.semantic_probabilities(side_features)
            side_analog = side_memory.query_many(model_features.loc[idx])
            meta_x = MetaPolicy.frame(
                calibration_oos.loc[idx],
                side_features,
                side_regime,
                side_analog,
                regime_persistence=side_persistence,
                regime_probs=side_regime_probabilities,
            )
            meta_model.fit(
                meta_x.loc[valid],
                pd.Series(labels_meta[valid], index=idx[valid]),
            )

        fit_side_meta(long_cal_oos, target_ret, self.meta, meta_memory, base_cost)
        fit_side_meta(
            short_cal_oos,
            short_target_ret,
            self.short_meta,
            short_meta_memory,
            short_cost,
        )

        base = self.model.predict(model_features)
        short_base = self.short_model.predict(model_features)
        meta_detector = self.meta_regimes if self.meta_regimes is not None else self.regimes
        meta_regime = meta_detector.transform(features)
        meta_persistence = meta_detector.persistence(features)
        meta_probabilities = meta_detector.semantic_probabilities(features)
        meta_x = MetaPolicy.frame(
            base,
            features,
            meta_regime,
            analog,
            regime_persistence=meta_persistence,
            regime_probs=meta_probabilities,
        )
        short_meta_x = MetaPolicy.frame(
            short_base,
            features,
            meta_regime,
            short_analog,
            regime_persistence=meta_persistence,
            regime_probs=meta_probabilities,
        )
        meta_all = self.meta.predict_proba(meta_x)
        short_meta_all = self.short_meta.predict_proba(short_meta_x)
        actions, scores = decide_actions(
            base,
            regime,
            analog,
            meta_all,
            self.settings,
            regime_persistence=regime_persistence,
            regime_probs=regime_probabilities,
            short_pred=short_base,
            short_meta_p=short_meta_all,
            short_analog=short_analog,
        )
        predictions = _assemble_predictions(
            base,
            short_base,
            regime,
            analog,
            short_analog,
            regime_persistence,
            meta_all,
            short_meta_all,
            actions,
            scores,
        )
        predictions.attrs["policy_diagnostics"] = dict(scores.attrs.get("diagnostics", {}))
        predictions.attrs["action_counts"] = {
            str(key): int(value) for key, value in actions.value_counts().items()
        }
        labels = labels.copy()
        labels["short_tb_label"] = short_labels["tb_label"]
        labels["short_tb_return"] = short_labels["tb_return"]
        return EngineArtifacts(
            features=features,
            labels=labels,
            regimes=regime,
            model=self.model,
            short_model=self.short_model,
            memory=self.memory,
            short_memory=self.short_memory,
            meta=self.meta,
            short_meta=self.short_meta,
            predictions=predictions,
        )

    @classmethod
    def load_bundle(cls, settings, out_dir):
        path = Path(out_dir)
        required = (
            "signal_model.joblib",
            "short_signal_model.joblib",
            "analog_memory.joblib",
            "short_analog_memory.joblib",
            "regime_detector.joblib",
            "meta_policy.joblib",
            "short_meta_policy.joblib",
        )
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete model bundle in {path}: {', '.join(missing)}"
            )
        # joblib uses pickle serialization: only load bundles produced by this
        # repository or another explicitly trusted training environment.
        import joblib

        obj = cls(settings)
        obj.model = SignalModel.load(path / "signal_model.joblib")
        obj.short_model = SignalModel.load(path / "short_signal_model.joblib")
        obj.memory = AnalogMemory.load(path / "analog_memory.joblib")
        obj.short_memory = AnalogMemory.load(path / "short_analog_memory.joblib")
        obj.memory.exclusion_bars = int(
            getattr(settings, "memory_exclusion_bars", max(1, getattr(settings, "validation_purge_bars", settings.horizon_bars)))
        )
        obj.short_memory.exclusion_bars = obj.memory.exclusion_bars
        obj.regimes = joblib.load(path / "regime_detector.joblib")
        obj.meta_regimes = (
            joblib.load(path / "meta_regime_detector.joblib")
            if (path / "meta_regime_detector.joblib").exists()
            else obj.regimes
        )
        obj.meta = joblib.load(path / "meta_policy.joblib")
        obj.short_meta = joblib.load(path / "short_meta_policy.joblib")
        efficiency_path = path / "feature_efficiency.joblib"
        obj.feature_efficiency = joblib.load(efficiency_path) if efficiency_path.exists() else None
        return obj

    def predict_frame(self, features, *, strict=True):
        if features is None or features.empty:
            return pd.DataFrame(index=getattr(features, "index", None))
        if strict:
            feature_cols = list(getattr(self.model, "feature_cols", ()) or ())
            if not feature_cols:
                raise ValueError("missing_live_feature_schema")
            absent = [column for column in feature_cols if column not in features.columns]
            if absent:
                raise ValueError("missing_live_feature_columns:" + ",".join(absent))
            last = features.reindex(columns=feature_cols).iloc[-1]
            numeric = pd.to_numeric(last, errors="coerce").to_numpy(dtype=float)
            missing_mask = ~np.isfinite(numeric)
            missing = [feature_cols[i] for i, flag in enumerate(missing_mask) if bool(flag)]
            missing_fraction = float(np.mean(missing_mask)) if len(missing_mask) else 1.0
            try:
                allowed_missing_fraction = float(
                    getattr(self.settings, "live_max_missing_feature_fraction", 0.0)
                )
            except (TypeError, ValueError, OverflowError):
                allowed_missing_fraction = -1.0
            if not np.isfinite(allowed_missing_fraction) or not 0.0 <= allowed_missing_fraction <= 1.0:
                raise ValueError("invalid_live_max_missing_feature_fraction")
            if missing_fraction > allowed_missing_fraction + 1e-12:
                raise ValueError(
                    f"live_feature_missing_fraction:{missing_fraction:.6f}>{allowed_missing_fraction:.6f}:"
                    + ",".join(missing[:32])
                )

        base = self.model.predict(features)
        short_base = self.short_model.predict(features)
        regime = self.regimes.transform(features)
        persistence = self.regimes.persistence(features)
        regime_probabilities = self.regimes.semantic_probabilities(features)
        analog = self.memory.query_many(features)
        short_analog = self.short_memory.query_many(features)
        meta_detector = self.meta_regimes if self.meta_regimes is not None else self.regimes
        meta_regime = meta_detector.transform(features)
        meta_persistence = meta_detector.persistence(features)
        meta_probabilities = meta_detector.semantic_probabilities(features)
        meta_x = MetaPolicy.frame(
            base,
            features,
            meta_regime,
            analog,
            regime_persistence=meta_persistence,
            regime_probs=meta_probabilities,
        )
        short_meta_x = MetaPolicy.frame(
            short_base,
            features,
            meta_regime,
            short_analog,
            regime_persistence=meta_persistence,
            regime_probs=meta_probabilities,
        )
        meta_p = self.meta.predict_proba(meta_x)
        short_meta_p = self.short_meta.predict_proba(short_meta_x)
        actions, scores = decide_actions(
            base,
            regime,
            analog,
            meta_p,
            self.settings,
            regime_persistence=persistence,
            regime_probs=regime_probabilities,
            short_pred=short_base,
            short_meta_p=short_meta_p,
            short_analog=short_analog,
        )
        predictions = _assemble_predictions(
            base,
            short_base,
            regime,
            analog,
            short_analog,
            persistence,
            meta_p,
            short_meta_p,
            actions,
            scores,
        )
        predictions.attrs["policy_diagnostics"] = dict(scores.attrs.get("diagnostics", {}))
        predictions.attrs["action_counts"] = {
            str(key): int(value) for key, value in actions.value_counts().items()
        }

        if strict:
            required = (
                "p_up",
                "p_short",
                "expected_return",
                "expected_return_lcb",
                "expected_return_ucb",
                "short_expected_return",
                "short_expected_return_lcb",
                "short_expected_return_ucb",
                "model_disagreement",
                "meta_success",
                "short_meta_success",
                "score",
                "short_score",
                "analog_edge",
                "analog_agreement",
                "analog_n",
                "short_analog_edge",
                "short_analog_agreement",
                "short_analog_n",
            )
            missing_outputs = [
                column
                for column in required
                if column not in predictions.columns or not pd.notna(predictions.iloc[-1][column])
            ]
            if missing_outputs:
                raise ValueError("missing_prediction_outputs:" + ",".join(missing_outputs))
            if (
                not pd.notna(predictions.iloc[-1].get("regime"))
                or not pd.notna(predictions.iloc[-1].get("regime_persistence"))
            ):
                raise ValueError("missing_prediction_regime")
        return predictions

    def save(self, out_dir):
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "signal_model.joblib")
        self.short_model.save(path / "short_signal_model.joblib")
        self.memory.save(path / "analog_memory.joblib")
        self.short_memory.save(path / "short_analog_memory.joblib")
        import joblib

        joblib.dump(self.regimes, path / "regime_detector.joblib")
        joblib.dump(
            self.meta_regimes if self.meta_regimes is not None else self.regimes,
            path / "meta_regime_detector.joblib",
        )
        joblib.dump(self.meta, path / "meta_policy.joblib")
        joblib.dump(self.short_meta, path / "short_meta_policy.joblib")
        joblib.dump(getattr(self, "feature_efficiency", None), path / "feature_efficiency.joblib")

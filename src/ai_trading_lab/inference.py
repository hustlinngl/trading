from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .deployment import bundle_compatibility, resolve_signal_bundle
from .features import make_features
from .memory import AnalogMemory
from .meta import MetaPolicy
from .models import SignalModel
from .policy import decide_actions


class InferenceBundleError(RuntimeError):
    """Fail-closed inference loading error with a stable reason code."""

    def __init__(self, reason: str, bundle: Path):
        self.reason = str(reason)
        self.bundle = Path(bundle)
        super().__init__(self.reason)


class InferenceBundle:
    """Inference-only view of a validated, side-specific deployment bundle.

    Both long and short predictions have independent probability models,
    outcome memories and meta-policies. This class intentionally owns no labels,
    fitting, optimization, promotion, persistence, or order placement.
    """

    def __init__(self, settings, bundle: Path):
        self.settings = settings
        self.bundle = Path(bundle)
        try:
            self.model = SignalModel.load(self.bundle / "signal_model.joblib")
            self.short_model = SignalModel.load(self.bundle / "short_signal_model.joblib")
            self.memory = AnalogMemory.load(self.bundle / "analog_memory.joblib")
            self.short_memory = AnalogMemory.load(self.bundle / "short_analog_memory.joblib")
            self.regimes = joblib.load(self.bundle / "regime_detector.joblib")
            meta_regime_path = self.bundle / "meta_regime_detector.joblib"
            self.meta_regimes = (
                joblib.load(meta_regime_path) if meta_regime_path.exists() else self.regimes
            )
            self.meta = joblib.load(self.bundle / "meta_policy.joblib")
            self.short_meta = joblib.load(self.bundle / "short_meta_policy.joblib")
            efficiency_path = self.bundle / "feature_efficiency.joblib"
            self.feature_efficiency = (
                joblib.load(efficiency_path) if efficiency_path.exists() else None
            )
        except Exception as exc:
            raise InferenceBundleError(
                f"bundle_load_error:{type(exc).__name__}", self.bundle
            ) from exc

        # joblib/pickle bundles can execute code at load time. Load only trusted,
        # locally produced artifacts whose manifest/hash was validated beforehand.
        exclusion = int(
            getattr(
                settings,
                "memory_exclusion_bars",
                max(1, getattr(settings, "validation_purge_bars", settings.horizon_bars)),
            )
        )
        self.memory.exclusion_bars = exclusion
        self.short_memory.exclusion_bars = exclusion

    @classmethod
    def load(cls, settings, root: str | Path, symbol: str) -> "InferenceBundle":
        bundle = resolve_signal_bundle(settings, root, symbol)
        if not (bundle / "signal_model.joblib").exists():
            raise InferenceBundleError("model_missing", bundle)

        compatible, reason = bundle_compatibility(settings, bundle, symbol)
        if not compatible:
            raise InferenceBundleError(reason, bundle)

        required = (
            "analog_memory.joblib",
            "short_signal_model.joblib",
            "short_analog_memory.joblib",
            "regime_detector.joblib",
            "meta_policy.joblib",
            "short_meta_policy.joblib",
        )
        missing = [name for name in required if not (bundle / name).exists()]
        if missing:
            raise InferenceBundleError(
                "deployment_bundle_incomplete:" + ",".join(missing), bundle
            )

        return cls(settings, bundle)

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build target-free canonical features for current closed market data."""
        features, _, _ = make_features(
            df,
            self.settings.horizon_bars,
            external_feature_lag_bars=getattr(
                self.settings, "external_feature_lag_bars", 1
            ),
        )
        return features

    def predict_frame(
        self, features: pd.DataFrame, *, strict: bool = True, profile: str = "strict"
    ) -> pd.DataFrame:
        profile = str(profile).strip().lower()
        if profile not in {"strict", "research"}:
            raise ValueError(f"unknown_signal_profile:{profile}")
        if features is None or features.empty:
            return pd.DataFrame(index=getattr(features, "index", None))

        if strict:
            feature_cols = list(getattr(self.model, "feature_cols", ()) or ())
            short_feature_cols = list(getattr(self.short_model, "feature_cols", ()) or ())
            if not feature_cols or not short_feature_cols:
                raise ValueError("missing_live_feature_schema")
            required_cols = sorted(set(feature_cols) | set(short_feature_cols))
            absent = [column for column in required_cols if column not in features.columns]
            if absent:
                raise ValueError("missing_live_features:" + ",".join(absent))
            last = features.reindex(columns=required_cols).iloc[-1]
            numeric = pd.to_numeric(last, errors="coerce").to_numpy(dtype=float)
            missing_mask = ~np.isfinite(numeric)
            missing = [required_cols[i] for i, flag in enumerate(missing_mask) if bool(flag)]
            missing_fraction = float(np.mean(missing_mask)) if len(missing_mask) else 1.0
            try:
                allowed_missing_fraction = float(
                    getattr(self.settings, "live_max_missing_feature_fraction", 0.0)
                )
            except (TypeError, ValueError, OverflowError):
                allowed_missing_fraction = -1.0
            if (
                not np.isfinite(allowed_missing_fraction)
                or not 0.0 <= allowed_missing_fraction <= 1.0
            ):
                raise ValueError("invalid_live_max_missing_feature_fraction")
            if missing_fraction > allowed_missing_fraction + 1e-12:
                raise ValueError(
                    f"live_feature_missing_fraction:{missing_fraction:.6f}>"
                    f"{allowed_missing_fraction:.6f}:" + ",".join(missing[:32])
                )

        base = self.model.predict(features)
        short_base = self.short_model.predict(features)
        regime = self.regimes.transform(features)
        persistence = self.regimes.persistence(features)
        regime_probs = self.regimes.semantic_probabilities(features)
        analog = self.memory.query_many(features)
        short_analog = self.short_memory.query_many(features)

        meta_detector = self.meta_regimes if self.meta_regimes is not None else self.regimes
        meta_regime = meta_detector.transform(features)
        meta_persistence = meta_detector.persistence(features)
        meta_probs = meta_detector.semantic_probabilities(features)

        meta_x = MetaPolicy.frame(
            base,
            features,
            meta_regime,
            analog,
            regime_persistence=meta_persistence,
            regime_probs=meta_probs,
        )
        short_meta_x = MetaPolicy.frame(
            short_base,
            features,
            meta_regime,
            short_analog,
            regime_persistence=meta_persistence,
            regime_probs=meta_probs,
        )
        meta_p = self.meta.predict_proba(meta_x)
        short_meta_p = self.short_meta.predict_proba(short_meta_x)
        policy_overrides = {}
        if profile == "research":
            policy_overrides = {
                "probability_threshold": float(getattr(
                    self.settings, "research_probability_threshold", 0.55
                )),
                "minimum_probability_edge": float(getattr(
                    self.settings, "research_minimum_probability_edge", 0.02
                )),
                "min_expected_return": float(getattr(
                    self.settings, "research_min_expected_return", 0.0005
                )),
                "decision_threshold": float(getattr(
                    self.settings, "research_decision_threshold", 0.05
                )),
                "meta_threshold": float(getattr(
                    self.settings, "research_meta_threshold", 0.50
                )),
                "uncertainty_penalty_mult": float(getattr(
                    self.settings, "research_uncertainty_penalty_mult", 1.0
                )),
            }
        actions, scores = decide_actions(
            base,
            regime,
            analog,
            meta_p,
            self.settings,
            regime_persistence=persistence,
            regime_probs=regime_probs,
            short_pred=short_base,
            short_meta_p=short_meta_p,
            short_analog=short_analog,
            **policy_overrides,
        )

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
        pred["regime_persistence"] = persistence
        pred["meta_success"] = meta_p
        pred["short_meta_success"] = short_meta_p
        pred["long_score"] = scores.attrs.get("long_scores", np.full(len(pred), np.nan))
        pred["short_score"] = scores.attrs.get("short_scores", np.full(len(pred), np.nan))
        pred["score"] = scores
        pred["action"] = actions
        pred["action_profile"] = profile
        pred = pred.replace([np.inf, -np.inf], np.nan)
        pred.attrs["policy_diagnostics"] = dict(scores.attrs.get("diagnostics", {}))
        pred.attrs["action_counts"] = {
            str(key): int(value) for key, value in actions.value_counts().items()
        }

        if strict:
            required_outputs = (
                "p_up",
                "probability_baseline",
                "p_short",
                "short_probability_baseline",
                "expected_return",
                "expected_return_lcb",
                "expected_return_ucb",
                "short_expected_return",
                "short_expected_return_lcb",
                "short_expected_return_ucb",
                "model_disagreement",
                "short_model_disagreement",
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
                for column in required_outputs
                if column not in pred.columns or not pd.notna(pred.iloc[-1][column])
            ]
            if missing_outputs:
                raise ValueError(
                    "missing_prediction_outputs:" + ",".join(missing_outputs)
                )
            if (
                not pd.notna(pred.iloc[-1].get("regime"))
                or not pd.notna(pred.iloc[-1].get("regime_persistence"))
            ):
                raise ValueError("missing_prediction_regime")

        return pred

    def predict(
        self, df: pd.DataFrame, *, strict: bool = True, profile: str = "strict"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build features and predict in one explicit inference-only call."""
        features = self.features(df)
        return features, self.predict_frame(features, strict=strict, profile=profile)

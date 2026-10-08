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
from .regimes import RegimeDetector


class InferenceBundleError(RuntimeError):
    """Fail-closed inference loading error with a stable reason code."""

    def __init__(self, reason: str, bundle: Path):
        self.reason = str(reason)
        self.bundle = Path(bundle)
        super().__init__(self.reason)


class InferenceBundle:
    """Inference-only view of a validated deployment bundle.

    This module deliberately owns no labels, fitting, optimization, promotion, or
    research persistence. It converts closed-candle observations into internal
    prediction rows that the live/paper adapters can compile into direct signals.
    """

    def __init__(self, settings, bundle: Path):
        self.settings = settings
        self.bundle = Path(bundle)
        try:
            self.model = SignalModel.load(self.bundle / "signal_model.joblib")
            self.memory = AnalogMemory.load(self.bundle / "analog_memory.joblib")
            self.regimes = joblib.load(self.bundle / "regime_detector.joblib")
            meta_regime_path = self.bundle / "meta_regime_detector.joblib"
            self.meta_regimes = (
                joblib.load(meta_regime_path) if meta_regime_path.exists() else self.regimes
            )
            self.meta = joblib.load(self.bundle / "meta_policy.joblib")
            try:
                self.feature_efficiency = joblib.load(
                    self.bundle / "feature_efficiency.joblib"
                )
            except Exception:
                self.feature_efficiency = None
        except Exception as exc:
            raise InferenceBundleError(
                f"bundle_load_error:{type(exc).__name__}", self.bundle
            ) from exc

        self.memory.exclusion_bars = int(
            getattr(
                settings,
                "memory_exclusion_bars",
                max(1, getattr(settings, "validation_purge_bars", settings.horizon_bars)),
            )
        )

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
            "regime_detector.joblib",
            "meta_policy.joblib",
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

    def predict_frame(self, features: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame(index=getattr(features, "index", None))

        if strict:
            last = features.iloc[-1]
            missing = [
                column
                for column in self.model.feature_cols
                if column not in features.columns or not pd.notna(last.get(column))
            ]
            if missing:
                raise ValueError("missing_live_features:" + ",".join(missing))

        base = self.model.predict(features)
        regime = self.regimes.transform(features)
        persistence = self.regimes.persistence(features)
        probs = self.regimes.semantic_probabilities(features)
        analog = self.memory.query_many(features)

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
        meta_p = self.meta.predict_proba(meta_x)
        actions, scores = decide_actions(
            base,
            regime,
            analog,
            meta_p,
            self.settings,
            regime_persistence=persistence,
            regime_probs=probs,
        )

        pred = base.copy()
        pred["regime"] = regime
        pred["analog_edge"] = analog.edge
        pred["analog_agreement"] = analog.agreement
        pred["analog_dispersion"] = analog.dispersion
        pred["analog_n"] = analog.n
        pred["regime_persistence"] = persistence
        pred["meta_success"] = meta_p
        pred["score"] = scores
        pred["action"] = actions
        pred = pred.replace([np.inf, -np.inf], np.nan)

        if strict:
            required = (
                "p_up",
                "expected_return",
                "expected_return_lcb",
                "expected_return_ucb",
                "model_disagreement",
                "meta_success",
                "score",
                "analog_edge",
                "analog_agreement",
                "analog_n",
            )
            missing_outputs = [
                column
                for column in required
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

    def predict(self, df: pd.DataFrame, *, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build features and predict in one explicit inference-only call."""
        features = self.features(df)
        return features, self.predict_frame(features, strict=strict)

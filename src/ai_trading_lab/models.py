from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False


@dataclass
class ModelReport:
    auc: float
    rmse: float
    brier: float
    n: int


class SignalModel:
    """Ensemble of heterogeneous learners with time-aware probability calibration."""
    def __init__(self, random_state: int = 42, xgb_estimators: int = 240, lgbm_estimators: int = 240, hist_max_iter: int = 260):
        self.random_state = random_state
        self.xgb_estimators = int(xgb_estimators)
        self.lgbm_estimators = int(lgbm_estimators)
        self.hist_max_iter = int(hist_max_iter)
        self.clfs = []
        self.regs = []
        if HAVE_XGB:
            self.clfs.append(XGBClassifier(n_estimators=self.xgb_estimators, max_depth=4, learning_rate=0.025, subsample=0.82, colsample_bytree=0.80, min_child_weight=5, reg_lambda=2.0, objective='binary:logistic', eval_metric='logloss', tree_method='hist', random_state=random_state, n_jobs=4))
            self.regs.append(XGBRegressor(n_estimators=self.xgb_estimators, max_depth=4, learning_rate=0.025, subsample=0.82, colsample_bytree=0.80, min_child_weight=5, reg_lambda=2.0, objective='reg:squarederror', tree_method='hist', random_state=random_state, n_jobs=4))
        if HAVE_LGBM:
            self.clfs.append(LGBMClassifier(n_estimators=self.lgbm_estimators, learning_rate=0.025, num_leaves=31, max_depth=-1, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=random_state, verbosity=-1))
            self.regs.append(LGBMRegressor(n_estimators=self.lgbm_estimators, learning_rate=0.025, num_leaves=31, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=random_state, verbosity=-1))
        self.clfs.append(HistGradientBoostingClassifier(max_iter=self.hist_max_iter, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=1.0, random_state=random_state))
        self.regs.append(HistGradientBoostingRegressor(max_iter=self.hist_max_iter, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=1.0, random_state=random_state))
        self.feature_cols = []
        self.fill_values = None
        self.weights = None
        self.weights_cls = None
        self.weights_reg = None
        self.calibrator = None
        self.calibrator_fallback = None
        self.calibration_oos_ = None
        self.fit_rows_ = 0
        self.conformal_abs_error_ = 0.0
        self.conformal_scaled_error_ = 0.0
        self.conformal_abs_residuals_ = None
        self.conformal_scaled_residuals_ = None
        self.conformal_scale_col = None
        self.conformal_level = 0.90
        self.ready = False

    def _clean(self, x: pd.DataFrame) -> pd.DataFrame:
        z = x[self.feature_cols].replace([np.inf, -np.inf], np.nan).copy()
        if self.fill_values is None:
            return z.fillna(0.0)
        return z.fillna(self.fill_values).fillna(0.0)

    def fit(self, X: pd.DataFrame, y_cls: pd.Series, y_ret: pd.Series, purge_bars: int = 0):
        self.feature_cols = [c for c in X.columns if X[c].dtype != 'object' and c not in {'target', 'future_ret'}]
        Xraw = X[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        mask = y_cls.notna() & y_ret.notna()
        Xraw = Xraw.loc[mask]
        yc = y_cls.loc[mask].astype(int)
        yr = y_ret.loc[mask].astype(float)
        if len(Xraw) < 100 or yc.nunique() < 2:
            raise ValueError('Insufficient training data or class diversity')

        n_cal = max(48, int(len(Xraw) * 0.15))
        use_cal = len(Xraw) > n_cal + 100
        if use_cal:
            purge = max(0, int(purge_bars))
            cal_start = len(Xraw) - n_cal
            core_end = max(0, cal_start - purge)
            if core_end < 100:
                use_cal = False
            else:
                Xcore_raw, Xcal_raw = Xraw.iloc[:core_end], Xraw.iloc[cal_start:]
                ycore_c, ycore_r = yc.iloc[:core_end], yr.iloc[:core_end]
                ycal = yc.iloc[cal_start:]
                ycal_r = yr.iloc[cal_start:]
                if ycore_c.nunique() < 2:
                    use_cal = False
        if use_cal:
            self.fill_values = Xcore_raw.median(numeric_only=True)
            Xcore = Xcore_raw.fillna(self.fill_values).ffill().fillna(0.0)
            Xcal = Xcal_raw.fillna(self.fill_values).ffill().fillna(0.0)
            for clf in self.clfs:
                clf.fit(Xcore, ycore_c)
            for reg in self.regs:
                reg.fit(Xcore, ycore_r)
            probs_cal = np.column_stack([m.predict_proba(Xcal)[:, 1] for m in self.clfs])
            rets_cal = np.column_stack([m.predict(Xcal) for m in self.regs])
            uniform = np.ones(len(self.clfs), dtype=float) / len(self.clfs)
            briers = np.array([brier_score_loss(ycal, probs_cal[:, j]) for j in range(probs_cal.shape[1])], dtype=float)
            rmses = np.array([mean_squared_error(ycal_r, rets_cal[:, j]) ** 0.5 for j in range(rets_cal.shape[1])], dtype=float)
            inv_b = 1.0 / np.maximum(briers, 1e-6); inv_r = 1.0 / np.maximum(rmses, 1e-8)
            learned_cls = inv_b / inv_b.sum(); learned_reg = inv_r / inv_r.sum()
            self.weights_cls = 0.5 * uniform + 0.5 * learned_cls
            self.weights_reg = 0.5 * uniform + 0.5 * learned_reg
            self.weights = uniform
            raw_p = probs_cal @ self.weights_cls
            calibration_frame = pd.DataFrame({'p_up_raw': raw_p}, index=Xcal.index)
            self.calibrator_fallback = None
            if ycal.nunique() < 2:
                self.calibrator = None
            elif len(Xcal) < 160 or np.unique(raw_p).size < 12:
                self.calibrator = LogisticRegression(C=1.0, solver='lbfgs', random_state=self.random_state).fit(calibration_frame, ycal)
            else:
                # Isotonic calibration is reliable only within the observed score
                # range. Full-data refitting can move deployment scores outside
                # that range; endpoint clipping would then turn extrapolated scores
                # into constant probabilities. Keep a Platt model, trained on the
                # same strictly OOS calibration scores, for extrapolation only.
                self.calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw_p, ycal)
                self.calibrator_fallback = LogisticRegression(
                    C=1.0, solver='lbfgs', random_state=self.random_state
                ).fit(calibration_frame, ycal)
            raw_er = rets_cal @ self.weights_reg
            residuals = np.abs(ycal_r.to_numpy(float) - raw_er)
            scale_col = 'atr_pct' if 'atr_pct' in Xcal.columns else ('vol_24' if 'vol_24' in Xcal.columns else None)
            self.conformal_scale_col = scale_col
            if len(residuals):
                self.conformal_abs_residuals_ = residuals.astype(float)
                self.conformal_abs_error_ = float(np.quantile(residuals, self.conformal_level, method='higher'))
                if scale_col is not None:
                    scale = np.maximum(np.abs(Xcal[scale_col].to_numpy(float)), 1e-6)
                    scaled = residuals / scale
                    self.conformal_scaled_residuals_ = scaled.astype(float)
                    self.conformal_scaled_error_ = float(np.quantile(scaled, self.conformal_level, method='higher'))
                else:
                    self.conformal_scaled_residuals_ = None
                    self.conformal_scaled_error_ = 0.0
            self.calibration_oos_ = pd.DataFrame({'p_up': raw_p, 'expected_return': raw_er, 'model_disagreement': probs_cal.std(axis=1), 'return_disagreement': rets_cal.std(axis=1)}, index=Xcal.index)
            # Calibration remains strictly OOS. Once calibration evidence is frozen,
            # refit the production learners on the full chronological training dataset so
            # deployment uses every available labeled row without contaminating validation.
            Xfit = Xraw.fillna(self.fill_values).ffill().fillna(0.0)
            self.fit_rows_ = int(len(Xfit))
            for clf in self.clfs: clf.fit(Xfit, yc)
            for reg in self.regs: reg.fit(Xfit, yr)
        else:
            self.fill_values = Xraw.median(numeric_only=True)
            Xfit = Xraw.fillna(self.fill_values).ffill().fillna(0.0)
            for clf in self.clfs: clf.fit(Xfit, yc)
            for reg in self.regs: reg.fit(Xfit, yr)
            self.fit_rows_ = int(len(Xfit))
            self.calibrator = None; self.calibrator_fallback = None; self.calibration_oos_ = None
            self.weights_cls = np.ones(len(self.clfs), dtype=float) / len(self.clfs)
            self.weights_reg = self.weights_cls.copy()
            self.conformal_abs_error_ = 0.0; self.conformal_scaled_error_ = 0.0
            self.conformal_abs_residuals_ = None; self.conformal_scaled_residuals_ = None; self.conformal_scale_col = None
        if self.weights_cls is None: self.weights_cls = np.ones(len(self.clfs), dtype=float) / len(self.clfs)
        if self.weights_reg is None: self.weights_reg = np.ones(len(self.regs), dtype=float) / len(self.regs)
        self.weights = np.ones(len(self.clfs), dtype=float) / len(self.clfs)
        self.ready = True
        return self

    def set_conformal_level(self, level: float):
        self.conformal_level = float(np.clip(level, 0.50, 0.999))
        if self.conformal_abs_residuals_ is not None and len(self.conformal_abs_residuals_):
            self.conformal_abs_error_ = float(np.quantile(self.conformal_abs_residuals_, self.conformal_level, method='higher'))
        if self.conformal_scaled_residuals_ is not None and len(self.conformal_scaled_residuals_):
            self.conformal_scaled_error_ = float(np.quantile(self.conformal_scaled_residuals_, self.conformal_level, method='higher'))
        return self

    def _predict_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.clfs])
        rets = np.column_stack([m.predict(X) for m in self.regs])
        w_cls = self.weights_cls if self.weights_cls is not None else np.ones(probs.shape[1]) / probs.shape[1]
        w_reg = self.weights_reg if self.weights_reg is not None else np.ones(rets.shape[1]) / rets.shape[1]
        return pd.DataFrame({'p_up_raw': probs @ w_cls, 'expected_return': rets @ w_reg}, index=X.index)

    def _calibrated_probabilities(self, raw_p) -> np.ndarray:
        """Calibrate raw probabilities, avoiding isotonic endpoint clipping off-support."""
        values = np.asarray(raw_p, dtype=float).reshape(-1)
        if self.calibrator is None:
            return np.clip(values, 0.0, 1.0)
        if isinstance(self.calibrator, LogisticRegression):
            frame = pd.DataFrame({"p_up_raw": values})
            return np.clip(self.calibrator.predict_proba(frame)[:, 1], 0.0, 1.0)

        calibrated = np.asarray(self.calibrator.predict(values), dtype=float)
        fallback = getattr(self, "calibrator_fallback", None)
        thresholds = getattr(self.calibrator, "X_thresholds_", None)
        if fallback is not None and thresholds is not None and len(thresholds):
            outside = (values < float(thresholds[0])) | (values > float(thresholds[-1]))
            if outside.any():
                frame = pd.DataFrame({"p_up_raw": values[outside]})
                calibrated[outside] = fallback.predict_proba(frame)[:, 1]
        return np.clip(calibrated, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.ready: raise RuntimeError('Model not fitted')
        Xn = self._clean(X); raw = self._predict_raw(Xn)
        raw['p_up'] = self._calibrated_probabilities(raw['p_up_raw'].to_numpy(float))
        pmat = np.column_stack([m.predict_proba(Xn)[:, 1] for m in self.clfs])
        rmat = np.column_stack([m.predict(Xn) for m in self.regs])
        raw['model_disagreement'] = pmat.std(axis=1)
        raw['return_disagreement'] = rmat.std(axis=1)
        if self.conformal_scale_col is not None and self.conformal_scaled_error_ > 0 and self.conformal_scale_col in X.columns:
            scale = np.maximum(np.abs(pd.to_numeric(X[self.conformal_scale_col], errors='coerce').to_numpy(float)), 1e-6)
            margin_arr = scale * float(self.conformal_scaled_error_)
        else:
            margin_arr = np.full(len(raw), float(max(0.0, self.conformal_abs_error_)))
        raw['expected_return_lcb'] = raw['expected_return'] - margin_arr
        raw['expected_return_ucb'] = raw['expected_return'] + margin_arr
        return raw[['p_up','expected_return','expected_return_lcb','expected_return_ucb','model_disagreement','return_disagreement']]

    def evaluate(self, X: pd.DataFrame, y_cls: pd.Series, y_ret: pd.Series) -> ModelReport:
        mask = y_cls.notna() & y_ret.notna(); pred = self.predict(X.loc[mask])
        auc = roc_auc_score(y_cls.loc[mask], pred['p_up']) if y_cls.loc[mask].nunique() > 1 else 0.5
        rmse = float(mean_squared_error(y_ret.loc[mask], pred['expected_return']) ** 0.5)
        brier = float(brier_score_loss(y_cls.loc[mask], pred['p_up']))
        return ModelReport(float(auc), rmse, brier, int(mask.sum()))

    def save(self, path: str | Path):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self, p)

    @staticmethod
    def load(path: str | Path) -> 'SignalModel':
        return joblib.load(path)

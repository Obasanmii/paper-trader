"""A walk-forward machine-learning strategy with explicit leakage controls."""
from __future__ import annotations

import numpy as np
import pandas as pd

from papertrader.strategies.base import Strategy, StrategyOutput, register


def _zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=window).mean()
    std = frame.rolling(window, min_periods=window).std()
    return (frame - mean) / std.replace(0.0, np.nan)


@register
class MLClassifier(Strategy):
    """Predict whether each symbol's next `horizon`-day return is positive; go long when confident.

    The model is pooled across symbols and retrained every `retrain_every` days
    on a rolling `train_window`. Leakage controls:

    * features at day t use bars <= t only;
    * retrain dates are fixed positions counted from the start of the data,
      so they don't depend on how much data comes after;
    * purging + embargo: a training sample dated s is only used at retrain
      date r once its label is known, i.e. s + horizon + embargo <= r;
    * the model trained after the close of r is first used for day r's
      decision, which is executed at the open of r+1.

    Position: a 1/N slice of capital when P(up) > threshold, flat otherwise.
    """

    name = "ml_classifier"
    FEATURES = ("ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "dist_sma50", "range_10", "volume_z20")

    def __init__(
        self,
        horizon: int = 5,
        train_window: int = 756,
        retrain_every: int = 63,
        embargo: int = 5,
        threshold: float = 0.55,
        model: str = "logistic",
        min_train_samples: int = 500,
        warmup: int = 252,
        random_state: int = 0,
    ):
        if model not in ("logistic", "gbm"):
            raise ValueError("model must be 'logistic' or 'gbm'")
        if horizon < 1 or retrain_every < 1 or embargo < 0:
            raise ValueError("horizon and retrain_every must be >= 1, embargo >= 0")
        if warmup < horizon + embargo + 60:
            raise ValueError("warmup is too short for the feature lookbacks plus the label horizon")
        if not 0.5 <= threshold < 1:
            raise ValueError("threshold must be in [0.5, 1)")
        super().__init__(
            horizon=horizon,
            train_window=train_window,
            retrain_every=retrain_every,
            embargo=embargo,
            threshold=threshold,
            model=model,
            min_train_samples=min_train_samples,
            warmup=warmup,
            random_state=random_state,
        )
        self.horizon, self.train_window, self.retrain_every = int(horizon), int(train_window), int(retrain_every)
        self.embargo, self.threshold, self.model_name = int(embargo), float(threshold), model
        self.min_train_samples, self.warmup, self.random_state = int(min_train_samples), int(warmup), random_state

    def features(self, data) -> np.ndarray:
        """Returns an array of shape (days, symbols, features); every value uses bars <= that day."""
        close, high, low, volume = data.close, data.high, data.low, data.volume
        logc = np.log(close)
        r1 = logc.diff()
        frames = [
            r1,
            logc.diff(5),
            logc.diff(20),
            logc.diff(60),
            r1.rolling(20, min_periods=20).std(),
            close / close.rolling(50, min_periods=50).mean() - 1.0,
            ((high - low) / close).rolling(10, min_periods=10).mean(),
            _zscore(np.log1p(volume), 20),
        ]
        return np.stack([f.to_numpy(dtype=float) for f in frames], axis=-1)

    def _new_model(self):
        if self.model_name == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=self.random_state
        )

    def run(self, data):
        T, N = len(data.dates), len(data.symbols)
        X = self.features(data)
        K = X.shape[-1]
        logc = np.log(data.close.to_numpy(dtype=float))
        h = self.horizon
        fwd = np.full((T, N), np.nan)
        if T > h:
            fwd[:-h] = logc[h:] - logc[:-h]  # label for day s: known only at the close of s + h
        y = np.where(np.isfinite(fwd), (fwd > 0).astype(float), np.nan)

        proba = np.full((T, N), np.nan)
        for r in range(self.warmup, T, self.retrain_every):
            last_sample = r - h - self.embargo  # purge + embargo
            first_sample = max(0, r - self.train_window)
            if last_sample < first_sample:
                continue
            Xs = X[first_sample : last_sample + 1].reshape(-1, K)
            ys = y[first_sample : last_sample + 1].reshape(-1)
            ok = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
            if ok.sum() < self.min_train_samples or np.unique(ys[ok]).size < 2:
                continue
            model = self._new_model().fit(Xs[ok], ys[ok])
            end = min(T, r + self.retrain_every)
            Xp = X[r:end].reshape(-1, K)
            usable = np.isfinite(Xp).all(axis=1)
            p = np.full(Xp.shape[0], np.nan)
            if usable.any():
                p[usable] = model.predict_proba(Xp[usable])[:, 1]
            proba[r:end] = p.reshape(end - r, N)

        proba_df = pd.DataFrame(proba, index=data.dates, columns=data.symbols)
        weights = (proba_df > self.threshold).astype(float) / N
        return StrategyOutput(self.finalise(weights, data), proba_df)

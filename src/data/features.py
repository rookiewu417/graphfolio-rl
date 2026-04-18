"""Feature engineering: technical indicators + normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands


_INDICATOR_COLS = [
    "log_return", "volume_change",
    "macd", "macd_signal",
    "rsi_14", "atr_14",
    "bbands_pct_b", "momentum_20",
]


def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df: single-symbol DataFrame with columns [open, high, low, close, volume].
    Returns DataFrame with technical indicator columns.
    """
    df = price_df.copy().sort_index()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    df["log_return"] = np.log(close / close.shift(1))
    df["volume_change"] = vol.pct_change()

    macd_obj = MACD(close)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    df["rsi_14"] = RSIIndicator(close, window=14).rsi()
    df["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["bbands_pct_b"] = bb.bollinger_pband()

    df["momentum_20"] = ROCIndicator(close, window=20).roc()

    return df[_INDICATOR_COLS].dropna()


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    data: MultiIndex (date, symbol) DataFrame with OHLCV.
    Returns MultiIndex (date, symbol) DataFrame of normalized features.
    """
    symbols = data.index.get_level_values("symbol").unique()
    frames = []
    for sym in symbols:
        sym_df = data.xs(sym, level="symbol")[["open", "high", "low", "close", "volume"]]
        feat_df = compute_indicators(sym_df)
        feat_df = feat_df.assign(symbol=sym).set_index("symbol", append=True)
        frames.append(feat_df)

    features = pd.concat(frames).sort_index()
    features = _normalize(features)
    return features


def _normalize(df: pd.DataFrame, method: str = "z_score") -> pd.DataFrame:
    if method == "z_score":
        return df.groupby(level="date").transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    elif method == "minmax":
        return df.groupby(level="date").transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
    return df


def make_lookback_tensor(
    features: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    window: int,
) -> tuple[np.ndarray, list]:
    """
    Build tensor of shape (T, N, F) using last-day features (no window dim to keep it simple).
    For RNN-style window, extend this to (T, N, W, F).
    Returns (tensor, valid_dates).
    """
    feat_unstacked = features.unstack("symbol")
    # Reorder columns to match symbols order
    feat_unstacked = feat_unstacked.reindex(symbols, axis=1, level=1)

    valid_dates = [d for d in dates if d in feat_unstacked.index]
    T = len(valid_dates)
    N = len(symbols)
    F = len(_INDICATOR_COLS)

    tensor = np.zeros((T, N, F), dtype=np.float32)
    for t, date in enumerate(valid_dates):
        row = feat_unstacked.loc[date]
        # row has MultiIndex columns: (indicator, symbol)
        for f_idx, feat in enumerate(_INDICATOR_COLS):
            if (feat, ) in row.index or feat in [c[0] for c in row.index]:
                vals = row[feat].reindex(symbols).fillna(0.0).values
                tensor[t, :, f_idx] = vals.astype(np.float32)

    return tensor, valid_dates

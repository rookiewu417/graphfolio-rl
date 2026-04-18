"""Feature engineering: technical indicators + normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


_INDICATOR_COLS = [
    "log_return", "volume_change",
    "macd", "macd_signal",
    "rsi_14", "atr_14",
    "bbands_pct_b", "momentum_20",
]


def compute_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    price_df: single-symbol DataFrame with columns [open, high, low, close, volume].
    Returns DataFrame with technical indicator columns appended.
    """
    df = price_df.copy().sort_index()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_change"] = df["volume"].pct_change()

    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    bb = ta.bbands(df["close"], length=20)
    df["bbands_pct_b"] = bb["BBP_5_2.0"] if bb is not None and "BBP_5_2.0" in bb.columns else np.nan

    df["momentum_20"] = ta.mom(df["close"], length=20)

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
    """Cross-sectional z-score normalization per date."""
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
) -> np.ndarray:
    """
    Build tensor of shape (T, N, W, F):
      T = number of dates, N = assets, W = lookback window, F = features.
    Only includes dates where all W prior days have data.
    """
    feat_arr = features.unstack("symbol").reindex(symbols, axis=1, level=1)
    valid_dates = [d for d in dates if feat_arr.index.get_loc(d) >= window]
    T, N, F = len(valid_dates), len(symbols), len(_INDICATOR_COLS)
    tensor = np.zeros((T, N, window, F), dtype=np.float32)
    for t, date in enumerate(valid_dates):
        loc = feat_arr.index.get_loc(date)
        window_slice = feat_arr.iloc[loc - window + 1 : loc + 1]
        tensor[t] = window_slice.values.reshape(window, N, F).transpose(1, 0, 2)
    return tensor, valid_dates

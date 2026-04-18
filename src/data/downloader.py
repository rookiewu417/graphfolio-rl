"""Download and cache daily OHLCV data for the asset universe."""
import os
import time
import logging
from pathlib import Path

import pandas as pd
import tushare as ts
import akshare as ak
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_tushare_pro() -> ts.pro_api:
    token = os.environ["TUSHARE_TOKEN"]
    ts.set_token(token)
    return ts.pro_api()


def download_tushare(
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Download daily OHLCV via tushare. Returns MultiIndex (date, symbol) DataFrame."""
    cache_path = cache_dir / f"ohlcv_{start_date}_{end_date}.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    pro = _get_tushare_pro()
    frames = []
    for sym in tqdm(symbols, desc="Downloading"):
        try:
            df = pro.daily(
                ts_code=sym,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.rename(columns={"ts_code": "symbol", "trade_date": "date", "vol": "volume"})
            frames.append(df)
            time.sleep(0.05)  # respect rate limit
        except Exception as e:
            logger.warning(f"Failed to download {sym}: {e}")

    data = pd.concat(frames).set_index(["date", "symbol"]).sort_index()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_path)
    logger.info(f"Saved {len(data)} rows to {cache_path}")
    return data


def load_data(cfg) -> pd.DataFrame:
    """Load or download OHLCV data according to config."""
    cache_dir = Path(cfg.data.cache_dir)
    symbols = list(cfg.data.universe)
    return download_tushare(
        symbols=symbols,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
        cache_dir=cache_dir,
    )


def split_data(data: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based train/val/test split."""
    train = data.loc[:cfg.data.train_end]
    val = data.loc[cfg.data.train_end:cfg.data.val_end]
    test = data.loc[cfg.data.val_end:]
    return train, val, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.utils.config import load_config
    cfg = load_config()
    data = load_data(cfg)
    print(data.info())
    print(data.head())

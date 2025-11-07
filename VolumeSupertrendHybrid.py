# --- Volume-Adaptive SuperTrend Hybrid (Long + Short) ---
# by M.Aykut Ünal
# =============================================================

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
# =============================================================
#  Volume-Adaptive SuperTrend (VAST)
# =============================================================
def volume_supertrend(
    df: DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    vol_period: int = 20,
    vol_alpha: float = 0.7,
    vol_low: float = 0.5,
    vol_high: float = 2.0,
    flip_vol_gate: float = 1.0,
) -> DataFrame:

    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    volume = df['volume'].values.astype(float)

    n = len(df)
    if n == 0:
        return df

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values

    vol_ma = pd.Series(volume).rolling(vol_period, min_periods=1).mean().values
    vol_ma = np.where(vol_ma == 0, 1e-10, vol_ma)
    vol_ratio = np.clip(volume / vol_ma, vol_low, vol_high)

    eff_mult = multiplier * (1.0 / (vol_ratio ** vol_alpha))
    hl2 = (high + low) / 2.0

    upperband = hl2 + eff_mult * atr
    lowerband = hl2 - eff_mult * atr

    final_upper, final_lower = upperband.copy(), lowerband.copy()
    for i in range(1, n):
        if close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upperband[i]
        else:
            final_upper[i] = min(upperband[i], final_upper[i - 1])

        if close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lowerband[i]
        else:
            final_lower[i] = max(lowerband[i], final_lower[i - 1])

    supertrend = np.zeros(n)
    trend = np.ones(n, dtype=int)

    supertrend[0] = hl2[0]
    trend[0] = 1

    for i in range(1, n):
        prev_st = supertrend[i - 1]
        vr = vol_ratio[i]
        allow_flip = vr >= flip_vol_gate

        if prev_st == final_upper[i - 1]:
            if (close[i] > final_upper[i]) and allow_flip:
                supertrend[i] = final_lower[i]
                trend[i] = 1
            else:
                supertrend[i] = final_upper[i]
                trend[i] = -1
        else:
            if (close[i] < final_lower[i]) and allow_flip:
                supertrend[i] = final_upper[i]
                trend[i] = -1
            else:
                supertrend[i] = final_lower[i]
                trend[i] = 1

    df['vast'] = supertrend
    df['vast_trend'] = trend
    df['vast_vol_ratio'] = vol_ratio
    df['vast_eff_mult'] = eff_mult
    return df


# =============================================================
#  Freqtrade Strategy (Long + Short)
# =============================================================
class VolumeSupertrendHybrid(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count = 50

    # ROI & Stop
    # ROI table:
    minimal_roi = {
        "0": 0.66,
        "444": 0.234,
        "1090": 0.085,
        "2327": 0
    }

    # Stoploss:
    stoploss = -0.23

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.017  # value loaded from strategy
    trailing_stop_positive_offset = 0.031  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    can_short: bool = True
    process_only_new_candles = True
    use_custom_stoploss = False

    # Optuna parametreleri (opsiyonel)
    xlevrage = IntParameter(2, 20, default=9, space='buy')
    vast_period = IntParameter(3, 20, default=16, space="buy")
    vast_mult = DecimalParameter(0.5, 5.0, default=4.7, decimals=1, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = volume_supertrend(
            dataframe,
            period=int(self.vast_period.value),
            multiplier=float(self.vast_mult.value)
        )
        return dataframe

    # === LONG ===
# === ENTRY ===
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Entry
        dataframe.loc[
            (dataframe['vast_trend'] == 1) &
            (dataframe['close'] > dataframe['vast']),
            ['enter_long', 'enter_tag']
        ] = (1, 'vast_long')

        # Short Entry
        dataframe.loc[
            (dataframe['vast_trend'] == -1) &
            (dataframe['close'] < dataframe['vast']),
            ['enter_short', 'enter_tag']
        ] = (1, 'vast_short')

        return dataframe


    # === EXIT ===
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long Exit (trend tersine dönmüş)
        dataframe.loc[
            (dataframe['vast_trend'] == -1) &
            (dataframe['close'] < dataframe['vast']),
            ['exit_long', 'exit_tag']
        ] = (1, 'vast_flip')

        # Short Exit (trend tersine dönmüş)
        dataframe.loc[
            (dataframe['vast_trend'] == 1) &
            (dataframe['close'] > dataframe['vast']),
            ['exit_short', 'exit_tag']
        ] = (1, 'vast_flip')

        return dataframe
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                    proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                    side: str, **kwargs) -> float:
                    
            # Kaldıraç değerini doğrudan Optuna parametresinden alıp döndürür.
            return self.xlevrage.value

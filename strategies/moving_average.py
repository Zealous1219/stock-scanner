"""
移动均线策略 (Moving Average Strategy)

策略说明:
    基于均线的综合筛选策略，结合趋势、动量、量价多个维度进行选股。

参数:
    min_data_length (int): 最小数据长度要求，默认60
    volume_threshold (float): 成交额门槛(元)，默认1e8 (1亿)
    volume_ratio (float): 成交量放大倍数，默认1.5
    ma_windows (list): 均线周期列表，默认[20, 60]
    recent_days (int): 最近N天用于高低点判断，默认5

使用示例:
    from strategies import load_strategy

    # 方式1: 从配置加载
    strategy = load_strategy('moving_average')

    # 方式2: 直接创建
    from strategies.moving_average import MovingAverageStrategy
    strategy = MovingAverageStrategy({
        'ma_windows': [20, 60],
        'volume_ratio': 1.5
    })

作者: AI Assistant
日期: 2026-03-28
"""

from typing import Dict, Any, List

import pandas as pd

from data_utils import get_completed_weekly_bars
from strategy_runtime import StrategyContext, StrategyResult
from strategies.base import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """
    移动均线策略

    采用5维度技术面筛选，必须同时满足以下所有条件：
    1. 周线趋势：周线收盘价在20周均线上方
    2. 均线多头：MA20 > MA60（黄金交叉）
    3. 价格位置：收盘价在MA20上方
    4. 动量结构：最近N天高低点逐步抬高
    5. 量价配合：成交量大于均量的N倍
    """

    DEFAULT_PARAMS = {
        'min_data_length': 60,
        'volume_threshold': 1e8,
        'volume_ratio': 1.5,
        'ma_windows': [20, 60],
        'recent_days': 5,
    }

    def __init__(self, params: Dict[str, Any] = None):
        """
        初始化移动均线策略

        参数:
            params: 策略参数字典，可包含以下键:
                - min_data_length: 最小数据长度
                - volume_threshold: 成交额门槛
                - volume_ratio: 成交量放大倍数
                - ma_windows: 均线周期列表
                - recent_days: 最近天数
        """
        super().__init__(params)

    def _init_strategy(self):
        """策略特定初始化"""
        self.min_data_length = self.params.get('min_data_length', 60)
        self.volume_threshold = self.params.get('volume_threshold', 1e8)
        self.volume_ratio = self.params.get('volume_ratio', 1.5)
        self.ma_windows = self.params.get('ma_windows', [20, 60])
        self.recent_days = self.params.get('recent_days', 5)

    def _validate_params(self):
        """验证策略参数"""
        if self.min_data_length < 20:
            raise ValueError("min_data_length must be >= 20")

        if self.volume_ratio < 1.0:
            raise ValueError("volume_ratio must be >= 1.0")

        if not isinstance(self.ma_windows, list) or len(self.ma_windows) < 2:
            raise ValueError("ma_windows must be a list with at least 2 elements")

        if self.recent_days < 2:
            raise ValueError("recent_days must be >= 2")

    def _convert_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将日线数据转换为周线数据

        参数:
            df: 日线数据

        返回值:
            周线数据
        """
        df = df.copy()
        df.set_index('date', inplace=True)

        weekly = pd.DataFrame()
        weekly['open'] = df['open'].resample('W').first()
        weekly['close'] = df['close'].resample('W').last()
        weekly['high'] = df['high'].resample('W').max()
        weekly['low'] = df['low'].resample('W').min()
        weekly['volume'] = df['volume'].resample('W').sum()

        weekly = weekly.dropna().reset_index()
        return weekly

    def _calculate_ma(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        计算简单移动平均线

        参数:
            df: 股票数据
            windows: 均线周期列表

        返回值:
            添加了MA列的数据
        """
        df = df.copy()
        for window in windows:
            df[f'ma{window}'] = df['close'].rolling(window=window).mean()
        return df

    def _weekly_filter(self, df: pd.DataFrame, now=None) -> bool:
        """
        周线趋势过滤

        参数:
            df: 日线数据

        返回值:
            True表示周线趋势向上
        """
        weekly = get_completed_weekly_bars(df, now=now)

        if len(weekly) < 20:
            return False

        weekly = self._calculate_ma(weekly, [20])
        latest = weekly.iloc[-1]

        return latest['close'] > latest['ma20']

    def _check_golden_cross(self, df: pd.DataFrame) -> bool:
        """
        检查均线黄金交叉

        参数:
            df: 已计算MA的数据

        返回值:
            True表示形成黄金交叉
        """
        latest = df.iloc[-1]
        return latest['ma20'] > latest['ma60']

    def _check_price_above_ma(self, df: pd.DataFrame, ma_col: str = 'ma20') -> bool:
        """
        检查价格是否在均线上方

        参数:
            df: 已计算MA的数据
            ma_col: 均线列名

        返回值:
            True表示价格在均线上方
        """
        latest = df.iloc[-1]
        return latest['close'] > latest[ma_col]

    def _check_higher_highs_lows(self, recent_df: pd.DataFrame) -> tuple:
        """
        检查高低点是否逐步抬高

        参数:
            recent_df: 最近N天数据

        返回值:
            (cond_highs, cond_lows) 元组
        """
        highs = recent_df['high'].values
        lows = recent_df['low'].values

        cond_highs = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
        cond_lows = all(lows[i] > lows[i-1] for i in range(1, len(lows)))

        return cond_highs, cond_lows

    def _check_volume_expansion(self, df: pd.DataFrame) -> bool:
        """
        检查成交量是否放大

        参数:
            df: 股票数据

        返回值:
            True表示成交量明显放大
        """
        latest = df.iloc[-1]
        return latest['volume'] > latest['vol_ma20'] * self.volume_ratio

    def compute(self, df: pd.DataFrame) -> bool:
        """
        计算策略信号

        综合判断股票是否满足所有筛选条件。

        参数:
            df: 股票历史数据

        返回值:
            True表示符合所有条件，False表示不符合
        """
        if len(df) < self.min_data_length:
            return False

        if not self._weekly_filter(df):
            return False

        df = self._calculate_ma(df, self.ma_windows)
        df['vol_ma20'] = df['volume'].rolling(20).mean()

        if not self._check_golden_cross(df):
            return False

        if not self._check_price_above_ma(df, 'ma20'):
            return False

        recent = df.iloc[-self.recent_days:]
        cond_highs, cond_lows = self._check_higher_highs_lows(recent)
        if not (cond_highs and cond_lows):
            return False

        if not self._check_volume_expansion(df):
            return False

        latest = df.iloc[-1]
        amount = latest['volume'] * latest['close']
        if amount < self.volume_threshold:
            return False

        return True

    def scan(self, symbol: str, df: pd.DataFrame, context: StrategyContext) -> StrategyResult:
        if len(df) < self.min_data_length:
            return StrategyResult(
                matched=False,
                reason_code="insufficient_daily_bars",
                reason_text="Not enough daily bars for the moving average strategy.",
            )

        if not self._weekly_filter(df, now=context.now):
            return StrategyResult(
                matched=False,
                reason_code="weekly_trend_failed",
                reason_text="The latest completed weekly trend filter did not pass.",
            )

        matched = self.compute(df)
        if not matched:
            return StrategyResult(
                matched=False,
                reason_code="rule_not_matched",
                reason_text="Symbol did not satisfy the moving average rules.",
            )

        latest = df.iloc[-1]
        return StrategyResult(
            matched=True,
            reason_code="matched",
            reason_text="Symbol matched the moving average strategy.",
            details={
                "signal_type": "moving_average_ready",
                "latest_date": str(pd.to_datetime(latest["date"]).date()),
                "close": float(latest["close"]),
            },
        )

    @property
    def supported_timeframes(self) -> List[str]:
        """
        支持的时间周期

        返回值:
            支持的时间周期列表
        """
        return ['daily', 'weekly']

# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np

from abc import abstractmethod
from typing import Dict, Union, List
from gym.spaces import Box

from tensortrade import Component
from tensortrade.trades import Trade
from tensortrade.features import FeaturePipeline

TypeString = Union[type, str]


class Exchange(Component):
    """
    An abstract exchange for use within a trading environments.
    合约交易账号抽象类，满足q-learning的evnironment要求

    Arguments:
        base_instrument: The exchange symbol of the instrument to store/measure value in.
        dtype: A type or str corresponding to the dtype of the `observation_space`.
        feature_pipeline: A pipeline of feature transformations for transforming observations.
    """
    registered_name = "exchanges"

    def __init__(self, dtype: TypeString = np.float32, feature_pipeline: FeaturePipeline = None, **kwargs):
        self._base_instrument = self.context.base_instrument
        self._dtype = self.default('dtype', dtype)
        self._feature_pipeline = self.default('feature_pipeline', feature_pipeline)
        self._window_size = self.default('window_size', 1, kwargs)
        self._min_trade_amount = self.default('min_trade_amount', 1e-6, kwargs)
        self._max_trade_amount = self.default('max_trade_amount', 1e6, kwargs)
        self._min_trade_price = self.default('min_trade_price', 1e-8, kwargs)
        self._max_trade_price = self.default('max_trade_price', 1e8, kwargs)

    @property
    def base_instrument(self) -> str:
        """
        对交易合约的基准衡量价值，如果股票/期货，为人民币/美金，如果数字货币，为USD/USDT
        The exchange symbol of the instrument to store/measure value in.
        """
        return self._base_instrument

    @base_instrument.setter
    def base_instrument(self, base_instrument: str):
        self._base_instrument = base_instrument

    @property
    def window_size(self) -> int:
        """The window size of observations."""
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        self._window_size = window_size

    @property
    def dtype(self) -> TypeString:
        """
        A type or str corresponding to the dtype of the `observation_space`.
        返回观测空间对应的类型或类型字符串描述
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: TypeString):
        self._dtype = dtype

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """
        A pipeline of feature transformations for transforming observations.
        获取特征库，用作观测数据转换为系列特征值
        """
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        self._feature_pipeline = feature_pipeline

    @property
    def base_precision(self) -> float:
        """The floating point precision of the base instrument."""
        return self._base_precision

    @base_precision.setter
    def base_precision(self, base_precision: float):
        """ 设置基准合约价格精度 """
        self._base_precision = base_precision

    @property
    def instrument_precision(self) -> float:
        """The floating point precision of the instrument to be traded."""
        return self._instrument_precision

    @instrument_precision.setter
    def instrument_precision(self, instrument_precision: float):
        """设置交易合约精度"""
        self._instrument_precision = instrument_precision

    @property
    @abstractmethod
    def initial_balance(self) -> float:
        """
        The initial balance of the base symbol on the exchange.
        期初净值（按照基准合约计算），例如10000个USD/USDT,或10000人民币
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def balance(self) -> float:
        """
        The current balance of the base symbol on the exchange.
        当前交易账号的净值（按照基准合约计算），例如3000个USD/USDT,或3000人民币
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def portfolio(self) -> Dict[str, float]:
        """
        The current balance of each symbol on the exchange (non-positive balances excluded).
        当前所有持仓品种净值列表
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def trades(self) -> List[Trade]:
        """A list of trades made on the exchange since the last reset."""
        raise NotImplementedError

    @property
    @abstractmethod
    def performance(self) -> pd.DataFrame:
        """The performance of the active account on the exchange since the last reset."""
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_columns(self) -> List[str]:
        """The final columns provided by the observation space, after any feature transformations."""
        raise NotImplementedError

    @property
    def observation_space(self) -> Box:
        """
        The final shape of the observations generated by the exchange, after any feature transformations.
        获取由交易账号和特征值运算后得出的观测数据维度空间
        """
        n_features = len(self.observation_columns)

        low = np.tile(self._min_trade_price, n_features)
        high = np.tile(self._max_trade_price, n_features)

        if self._window_size > 1:
            low = np.tile(low, self._window_size).reshape((self._window_size, n_features))
            high = np.tile(high, self._window_size).reshape((self._window_size, n_features))

        return Box(low=low, high=high, dtype=self._dtype)

    @property
    def net_worth(self) -> float:
        """Calculate the net worth of the active account on the exchange.

        Returns:
            The total portfolio value of the active account on the exchange.
        """
        net_worth = self.balance
        portfolio = self.portfolio

        if not portfolio:
            return net_worth

        for symbol, amount in portfolio.items():
            if symbol == self._base_instrument:
                continue

            current_price = self.current_price(symbol=symbol)
            net_worth += current_price * amount

        return net_worth

    @property
    def profit_loss_percent(self) -> float:
        """Calculate the percentage change in net worth since the last reset.

        Returns:
            The percentage change in net worth since the last reset.
        """
        return float(self.net_worth / self.initial_balance) * 100

    @property
    @abstractmethod
    def has_next_observation(self) -> bool:
        """If `False`, the exchange's data source has run out of observations.

        Resetting the exchange may be necessary to continue generating observations.

        Returns:
            Whether or not the specified instrument has a next observation.
        """
        raise NotImplementedError

    def _next_observation(self) -> Union[pd.DataFrame, np.ndarray]:
        raise NotImplementedError()

    def next_observation(self) -> np.ndarray:
        """Generate the next observation from the exchange.
        Returns:
            The next multi-dimensional list of observations.
        """
        observation = self._next_observation()

        if isinstance(observation, pd.DataFrame):
            observation = observation.fillna(0, axis=1)
            return observation.values

        return observation

    def instrument_balance(self, symbol: str) -> float:
        """The current balance of the specified symbol on the exchange, denoted in the base instrument.
            当前指定合约的持仓数量
        Arguments:
            symbol: The symbol to retrieve the balance of.

        Returns:
            The balance of the specified exchange symbol, denoted in the base instrument.
        """
        portfolio = self.portfolio

        if symbol in portfolio.keys():
            return portfolio[symbol]

        return 0

    @abstractmethod
    def current_price(self, symbol: str) -> float:
        """The current price of an instrument on the exchange, denoted in the base instrument.

        Arguments:
            symbol: The exchange symbol of the instrument to get the price for.

        Returns:
            The current price of the specified instrument, denoted in the base instrument.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_trade(self, trade: Trade) -> Trade:
        """Execute a trade on the exchange, accounting for slippage.

        Arguments:
            trade: The trade to execute.

        Returns:
            The filled trade.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the feature pipeline, initial balance, trades, performance, and any other temporary stateful data."""
        if self._feature_pipeline is not None:
            self.feature_pipeline.reset()

        self._observation_generator = self._create_observation_generator()
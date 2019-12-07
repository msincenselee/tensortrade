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
# limitations under the License
import numpy as np

from typing import Union
from gym.spaces import Discrete

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, TradeType


class DiscreteActionStrategy(ActionStrategy):
    """
    Simple discrete strategy, which calculates the trade amount as a fraction of the total balance.
    简单的离散交易动作策略，按照交易仓位百分比进行交易
    """

    def __init__(self, n_actions: int = 20, instrument_symbol: str = 'BTC', max_allowed_slippage_percent: float = 1.0):
        """
        Arguments:
            n_actions: The number of bins to divide the total balance by. Defaults to 20 (i.e. 1/20, 2/20, ..., 20/20). 将账号资金划分多少份
            instrument_symbol: The exchange symbol of the instrument being traded. Defaults to 'BTC'. 交易合约
            max_allowed_slippage: The maximum amount above the current price the strategy will pay for an instrument. Defaults to 1.0 (i.e. 1%). 最大滑点百分比
        """
        super().__init__(action_space=Discrete(n_actions), dtype=np.int64)
        # 账号资金划分数量
        self.n_actions = n_actions
        # 交易合约
        self.instrument_symbol = instrument_symbol
        # 最大滑点百分比
        self.max_allowed_slippage_percent = max_allowed_slippage_percent

    @property
    def dtype(self) -> DTypeString:
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        raise ValueError(
            'Cannot change the dtype of a `SimpleDiscreteStrategy` due to the requirements of `gym.spaces.Discrete` spaces. ')

    def get_trade(self, action: TradeActionUnion) -> Trade:
        """
        The trade type is determined by `action % len(TradeType)`, and the trade amount is determined by the multiplicity of the action.
        获取Trade对象，交易类型，由 action % len(TradeType)决定， 交易数量， 根据action 动作对应级别获取
        For example, 1 = LIMIT_BUY|0.25, 2 = MARKET_BUY|0.25 ;  6 = LIMIT_BUY|0.5, 7 = MARKET_BUY|0.5, etc.
        """
        # 每个交易动作，对应的操作仓位份数。 20/ 5 = 4
        n_splits = self.n_actions / len(TradeType)
        # 交易类型 0 ~4 % 5
        trade_type = TradeType(action % len(TradeType))
        # 交易仓位， action/len(TradeType) +1 => 交易数量份数( 1~5) => 乘以每一份
        trade_amount_percent = (int(action / len(TradeType)) + 1) * float(1 / n_splits)

        # 当前交易合约价格
        current_price = self._exchange.current_price(symbol=self.instrument_symbol)
        # 基准合约价格精度
        base_precision = self._exchange.base_precision
        # 交易合约价格精度
        instrument_precision = self._exchange.instrument_precision

        # 当前合约持仓数量
        amount = self._exchange.instrument_balance(self.instrument_symbol)
        # 当前价格
        price = current_price

        # 市价/限价买入
        if trade_type is TradeType.MARKET_BUY or trade_type is TradeType.LIMIT_BUY:
            # 滑点调整=》价格
            price_adjustment = 1 + (self.max_allowed_slippage_percent / 100)
            # 精度修正=》价格
            price = max(round(current_price * price_adjustment, base_precision), base_precision)
            #  账号基准净值 * 仓位比例 / 价格 =》 修正 =》 买入交易合约数量
            amount = round(self._exchange.balance * 0.99 *
                           trade_amount_percent / price, instrument_precision)
        # 市价卖出/限价卖出
        elif trade_type is TradeType.MARKET_SELL or trade_type is TradeType.LIMIT_SELL:
            #  滑点调整=》价格
            price_adjustment = 1 - (self.max_allowed_slippage_percent / 100)
            # 精度修正=》价格
            price = round(current_price * price_adjustment, base_precision)
            # 交易合约当前持仓数量
            amount_held = self._exchange.portfolio.get(self.instrument_symbol, 0)
            # 持仓数量 * 仓位比例 => 修正 =》 卖出交易合约数量
            amount = round(amount_held * trade_amount_percent, instrument_precision)

        # 交易合约，交易类型，交易数量，交易价格 =》 Trade 对象
        return Trade(self.instrument_symbol, trade_type, amount, price)

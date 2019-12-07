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
from gym.spaces import Box

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, TradeType


class ContinuousActionStrategy(ActionStrategy):
    """
    Simple continuous strategy, which calculates the trade amount as a fraction of the total balance.
    简单的交易动作策略，按照交易仓位百分比进行交易
    """

    def __init__(self, instrument_symbol: str = 'BTC', max_allowed_slippage_percent: float = 1.0, dtype: DTypeString = np.float16):
        """
        Arguments:
            instrument_symbol: The exchange symbol of the instrument being traded. Defaults to 'BTC'. 交易合约
            max_allowed_slippage: The maximum amount above the current price the strategy will pay for an instrument. Defaults to 1.0 (i.e. 1%). 最大滑点百分比
            dtype: A type or str corresponding to the dtype of the `action_space`. Defaults to `np.float16`. 交易动作
        """

        super().__init__(action_space=Box(0, 1, shape=(1, 1), dtype=dtype), dtype=dtype)

        # 交易合约
        self.instrument_symbol = instrument_symbol
        # 最大滑点
        self.max_allowed_slippage_percent = max_allowed_slippage_percent

    def get_trade(self, action: TradeActionUnion) -> Trade:
        """
        get a new Trade object. 根据action参数，获取一个新的Trade对象
        :param action: 交易动作类型，（tuple)
        :return:
        """
        # 动作类型(int)， 交易仓位（0~1）
        action_type, trade_amount_percent = action
        # 交易类型
        trade_type = TradeType(int(action_type * len(TradeType)))

        # 获取合约的当前价格
        current_price = self._exchange.current_price(symbol=self.instrument_symbol)
        # 获取基准合约价格精度（例如USDT)
        base_precision = self._exchange.base_precision
        # 获取交易合约价格精度(例如BTC)
        instrument_precision = self._exchange.instrument_precision

        # 当前持有的
        # 获取交易合约当前持仓数量
        amount = self._exchange.instrument_balance(self.instrument_symbol)
        price = current_price

        # 市价买入/限价买入
        if trade_type is TradeType.MARKET_BUY or trade_type is TradeType.LIMIT_BUY:
            # 滑点调整=》价格
            price_adjustment = 1 + (self.max_allowed_slippage_percent / 100)
            # 精度修正=》价格
            price = max(round(current_price * price_adjustment, base_precision), base_precision)
            # 账号基准净值 * 仓位比例 / 价格 =》 修正 =》 买入交易合约数量
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

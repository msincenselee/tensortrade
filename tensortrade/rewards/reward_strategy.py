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

import pandas as pd

from abc import ABCMeta, abstractmethod

from tensortrade.trades import Trade


class RewardStrategy(object, metaclass=ABCMeta):
    """
    抽象类，奖赏策略
    """

    def __init__(self):
        pass

    @property
    def exchange(self) -> 'InstrumentExchange':
        """
        The exchange being used by the current trading environment. Setting the exchange causes the strategy to reset.
        获取当前交易账号
        """
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: 'InstrumentExchange'):
        """
        设置当前交易账号
        :param exchange:
        :return:
        """
        self._exchange = exchange
        self.reset()

    def reset(self):
        """
        Optionally implementable method for resetting stateful strategies.
        [可选]重置方法
        """
        pass

    @abstractmethod
    def get_reward(self, current_step: int, trade: Trade) -> float:
        """
        获取奖赏
        Arguments:
            current_step: The environment's current timestep. 当前切片的步骤数
            trade: The trade executed and filled this timestep. 当前切片的交易结果

        Returns:
            A float corresponding to the benefit earned by the action taken this timestep. 根据当前交易结果，得出的奖赏数值
        """
        raise NotImplementedError()

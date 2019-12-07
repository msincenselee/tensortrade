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
import numpy as np

from typing import Callable

from tensortrade.rewards import RewardStrategy
from tensortrade.trades import TradeType, Trade


class RiskAdjustedReturnStrategy(RewardStrategy):
    """
    A reward strategy that rewards the agent for increasing its net worth, while penalizing more volatile strategies.
    风险回报类奖赏策略
    """

    def __init__(self, return_algorithm: str = 'sharpe', risk_free_rate: float = 0., target_returns: float = 0.):
        """
        Args:
            return_algorithm (optional): The risk-adjusted return metric to use. Options are 'sharpe' and 'sortino'. Defaults to 'sharpe'.
            risk_free_rate (optional): The risk free rate of returns to use for calculating metrics. Defaults to 0.
            target_returns (optional): The target returns per period for use in calculating the sortino ratio. Default to 0.
        """
        # 返回算法
        self._return_algorithm = self._return_algorithm_from_str(return_algorithm)
        # 风险度
        self._risk_free_rate = risk_free_rate
        # 目标回报
        self._target_returns = target_returns

    def _return_algorithm_from_str(self, algorithm_str: str) -> Callable[[pd.DataFrame], float]:
        """
        str =》 算法实例
        :param algorithm_str:
        :return:
        """
        # 使用夏普比率
        if algorithm_str is 'sharpe':
            return self._sharpe_ratio
        # 使用sortino
        elif algorithm_str is 'sortino':
            return self._sortino_ratio

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Return the sharpe ratio for a given series of a returns.
        返回夏普比率

        https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (returns.mean() - self._risk_free_rate) / (returns.std() + 1E-9)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """
        Return the sortino ratio for a given series of a returns.
        返回索提诺比率
        Sortino ratio的思路和Sharpe ratio的思路是一样的，但是对分子分母分别都做了调整。它将分子换为超额收益率，
        而分母换为Lower partial standard deviation,下偏标准差，主要是为了解决传统的正态分布存在的几个问题：
        分布其实并不对称。尤其是收益率函数分布左偏（偏度为负）的情况下，正态分布会低估风险，
        此时使用偏态分布就要比正态分布要合理；投资组合的下限应该是无风险投资工具。
        因此传统的sharpe ratio中分母使用全体的标准差（全体对平均投资收益的偏离）是不合适的，
        应该使用收益对无风险投资收益的偏离。总体上来看，Sortino ratio更看重对（左）尾部的预期损失分析，
        而Sharpe ratio则是对全体样本进行分析；而当  更换为收益列中大于无风险收益的全部样本时，
        则是对（右）尾部的超额收入分析。金融历史不能完全指导过去，尤其是在极端风险的考量上。
        因此使用Sortino ratio就变成一种更审慎的评估工具了。另外说一句，Sortino ratio也没有解决尖峰（leptokurtic）肥尾(fat tail)的问题。

        链接：https://www.zhihu.com/question/37128695/answer/230508370


        https://en.wikipedia.org/wiki/Sortino_ratio
        """
        # 亏损收益
        downside_returns = (returns[returns < self._target_returns])**2

        # 资产期望收益率
        expected_return = returns.mean()

        # 亏损收益方差
        downside_std = np.sqrt(downside_returns.mean())

        return (expected_return - self._risk_free_rate) / (downside_std + 1E-9)

    def get_reward(self, current_step: int, trade: Trade) -> float:
        """Return the reward corresponding to the selected risk-adjusted return metric."""
        returns = self._exchange.performance['net_worth'].diff()

        risk_adjusted_return = self._return_algorithm(returns)

        return risk_adjusted_return

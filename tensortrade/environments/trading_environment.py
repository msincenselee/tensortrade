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

import gym
import logging
import pandas as pd
import numpy as np
import tensortrade.exchanges as exchanges
import tensortrade.actions as actions
import tensortrade.rewards as rewards
import tensortrade.features as features

from gym import spaces
from typing import Union, Tuple, List

from tensortrade.actions import ActionStrategy, TradeActionUnion
from tensortrade.rewards import RewardStrategy
from tensortrade.exchanges import InstrumentExchange
from tensortrade.features import FeaturePipeline
from tensortrade.trades import Trade


class TradingEnvironment(gym.Env):
    """
    A trading environment made for use with Gym-compatible reinforcement learning algorithms.
    兼容GYM强化学习算法的的交易环境
    """

    def __init__(self,
                 exchange: Union[InstrumentExchange, str],
                 action_strategy: Union[ActionStrategy, str],
                 reward_strategy: Union[RewardStrategy, str],
                 feature_pipeline: Union[FeaturePipeline, str] = None,
                 **kwargs):
        """
        Arguments:
            exchange: The `InstrumentExchange` that will be used to feed data from and execute trades within.
            action_strategy:  The strategy for transforming an action into a `Trade` at each timestep.
            reward_strategy: The strategy for determining the reward at each timestep.
            feature_pipeline (optional): The pipeline of features to pass the observations through.
            kwargs (optional): Additional arguments for tuning the environment, logging, etc.
        """
        super().__init__()

        # str => exchange , 当前交易账号
        self._exchange = exchanges.get(exchange) if isinstance(exchange, str) else exchange

        # str => 交易动作策略
        self._action_strategy = actions.get(action_strategy) if isinstance(
            action_strategy, str) else action_strategy

        # str => 奖赏策略
        self._reward_strategy = rewards.get(reward_strategy) if isinstance(
            reward_strategy, str) else reward_strategy

        # str =》 特征
        self._feature_pipeline = features.get(feature_pipeline) if isinstance(
            feature_pipeline, str) else feature_pipeline

        # 使用交易账号的特征
        if feature_pipeline is not None:
            self._exchange.feature_pipeline = feature_pipeline

        # 重置交易账号
        self._exchange.reset()

        # 链接交易动作策略与交易账号
        self._action_strategy.exchange = self._exchange
        # 链接奖赏策略与交易账号
        self._reward_strategy.exchange = self._exchange

        # 观测维度空间
        self.observation_space = self._exchange.observation_space

        # 动作维度空间
        self.action_space = self._action_strategy.action_space

        # 日志记录
        self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
        self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

        # 取消tensorflow的日志记录
        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

    @property
    def exchange(self) -> InstrumentExchange:
        """
        The `InstrumentExchange` that will be used to feed data from and execute trades within.
        获取交易账号，它混合了数据灌输和交易过程
        """
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: InstrumentExchange):
        """
        设置交易账号
        :param exchange:
        :return:
        """
        self._exchange = exchange

    @property
    def action_strategy(self) -> ActionStrategy:
        """
        The strategy for transforming an action into a `Trade` at each timestep.
        获取交易动作策略，它根据每一个时间切片，将动作转换为交易指令
        """
        return self._action_strategy

    @action_strategy.setter
    def action_strategy(self, action_strategy: ActionStrategy):
        """
        set action_strategy instance.
        设置交易动作策略实例
        :param action_strategy:
        :return:
        """
        self._action_strategy = action_strategy

    @property
    def reward_strategy(self) -> RewardStrategy:
        """
        The strategy for determining the reward at each timestep.
        获取奖赏策略，它决定每一步操作带来的奖赏
        """
        return self._reward_strategy

    @reward_strategy.setter
    def reward_strategy(self, reward_strategy: RewardStrategy):
        """
        set reward startegy instance
        设置奖赏策略实例
        :param reward_strategy:
        :return:
        """
        self._reward_strategy = reward_strategy

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """
        The feature pipeline to pass the observations through.
        获取特征库
        :return:
        """
        return self._exchange.feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        """
        设置特征库
        :param feature_pipeline:
        :return:
        """
        self._exchange.feature_pipeline = feature_pipeline

    def _take_action(self, action: TradeActionUnion) -> Trade:
        """Determines a specific trade to be taken and executes it within the exchange.
        执行动作，进行交易
        Arguments:
            action: The trade action provided by the agent for this timestep.
            由智能体根据当前时间切片得出动作策略

        Returns:
            A tuple containing the (fill_amount, fill_price) of the executed trade.
        """
        executed_trade = self._action_strategy.get_trade(action=action)

        filled_trade = self._exchange.execute_trade(executed_trade)

        return executed_trade, filled_trade

    def _next_observation(self, trade: Trade) -> np.ndarray:
        """Returns the next observation from the exchange.
        获取下一步的观测值
        Returns:
            The observation provided by the environment's exchange, often OHLCV or tick trade history data points.
        """
        self._current_step += 1

        observation = self._exchange.next_observation()
        if len(observation) != 0:
            observation = observation[0]
            observation = np.nan_to_num(observation)
        return observation

    def _get_reward(self, trade: Trade) -> float:
        """Returns the reward for the current timestep.
        返回当前切片的奖赏
        Returns:
            A float corresponding to the benefit earned by the action taken this step.
            根据当前动作交易产生的利润
        """
        reward = self._reward_strategy.get_reward(current_step=self._current_step, trade=trade)
        reward = np.nan_to_num(reward)

        if np.bitwise_not(np.isfinite(reward)):
            raise ValueError('Reward returned by the reward strategy must by a finite float.')

        return reward

    def _done(self) -> bool:
        """Returns whether or not the environment is done and should be restarted.
        如果环境没有下一步观测空间，或者净值过低，就返回终结标志
        Returns:
            A boolean signaling whether the environment is done and should be restarted.
        """
        lost_90_percent_net_worth = self._exchange.profit_loss_percent < 0.1

        return lost_90_percent_net_worth or not self._exchange.has_next_observation

    def _info(self, executed_trade: Trade, filled_trade: Trade) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.
        获取当前数据，包括当前步骤，交易账号，执行交易，完成的交易
        Returns:
            info: A dictionary containing the exchange used, the current timestep, and the filled trade, if any.
        """
        return {'current_step': self._current_step,
                'exchange': self._exchange,
                'executed_trade': executed_trade,
                'filled_trade': filled_trade}

    def step(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environment based on the specified action.
        根据指定的动作，执行一步。
        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environment's exchange, often OHLCV or tick trade history data points. 下一观测空间
            reward (float): An amount corresponding to the benefit earned by the action taken this timestep. 奖赏值
            done (bool): If `True`, the environment is complete and should be restarted. 环境是否完结
            info (dict): Any auxiliary, diagnostic, or debugging information to output. 当前状态
        """
        executed_trade, filled_trade = self._take_action(action)

        observation = self._next_observation(filled_trade)
        reward = self._get_reward(filled_trade)
        done = self._done()
        info = self._info(executed_trade, filled_trade)

        return observation, reward, done, info

    def reset(self) -> pd.DataFrame:
        """Resets the state of the environment and returns an initial observation.
        重置
        Returns:
            observation: the initial observation. 初始的观测空间
        """
        # 重置交易动作策略
        self._action_strategy.reset()
        # 重置奖赏策略
        self._reward_strategy.reset()
        # 重置交易账号
        self._exchange.reset()
        # 重置当前步骤记录数
        self._current_step = 0

        return self._next_observation(Trade('N/A', 'hold', 0, 0))

    def render(self, mode='none'):
        """Renders the environment."""
        pass

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
import importlib
import pandas as pd
import numpy as np

import tensortrade.exchanges as exchanges
import tensortrade.actions as actions
import tensortrade.rewards as rewards
import tensortrade.features as features

from gym import spaces
from typing import Union, Tuple, List, Dict

from tensortrade.actions import ActionScheme, TradeActionUnion
from tensortrade.rewards import RewardScheme
from tensortrade.exchanges import Exchange
from tensortrade.features import FeaturePipeline
from tensortrade.trades import Trade

if importlib.util.find_spec("matplotlib") is not None:
    from tensortrade.environments.render import MatplotlibTradingChart


class TradingEnvironment(gym.Env):
    """
    A trading environments made for use with Gym-compatible reinforcement learning algorithms.
    兼容GYM强化学习算法的的交易环境
    """

    def __init__(self,
                 exchange: Union[Exchange, str],
                 action_scheme: Union[ActionScheme, str],
                 reward_scheme: Union[RewardScheme, str],
                 feature_pipeline: Union[FeaturePipeline, str] = None,
                 **kwargs):
        """
        Arguments:
            exchange: The `Exchange` that will be used to feed data from and execute trades within.
            action_scheme:  The component for transforming an action into a `Trade` at each timestep.
            reward_scheme: The component for determining the reward at each timestep.
            feature_pipeline (optional): The pipeline of features to pass the observations through.
            kwargs (optional): Additional arguments for tuning the environment, logging, etc.
        """
        super().__init__()
        # str => exchange , 当前交易账号
        self._exchange = exchanges.get(exchange) if isinstance(exchange, str) else exchange

        # str => 交易动作方案
        self._action_scheme = actions.get(action_scheme) if isinstance(
            action_scheme, str) else action_scheme

        # str => 奖赏方案
        self._reward_scheme = rewards.get(reward_scheme) if isinstance(
            reward_scheme, str) else reward_scheme
        # # str =》 特征转换管道
        self._feature_pipeline = features.get(feature_pipeline) if isinstance(
            feature_pipeline, str) else feature_pipeline

        if feature_pipeline is not None:
            self._exchange.feature_pipeline = feature_pipeline

        # 链接交易动作方案与交易账号
        self._action_scheme.exchange = self._exchange
        # 链接奖赏方案与交易账号
        self._reward_scheme.exchange = self._exchange

        # 观测维度空间
        self.observation_space = self._exchange.observation_space

        # 动作维度空间
        self.action_space = self._action_scheme.action_space

        self.render_benchmarks: List[Dict] = kwargs.get('render_benchmarks', [])
        self.viewer = None

        self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
        self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))
        # 取消tensorflow的日志记录
        logging.getLogger('tensorflow').disabled = kwargs.get('disable_tensorflow_logger', True)

        self.reset()

    @property
    def exchange(self) -> Exchange:
        """
        The `Exchange` that will be used to feed data from and execute trades within.
        获取交易账号，它混合了数据灌输和交易过程
        """
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: Exchange):
        """设置交易账号"""
        self._exchange = exchange

    @property
    def episode_trades(self) -> pd.DataFrame:
        """A `pandas.DataFrame` of trades made this episode."""
        return self.exchange.trades

    @property
    def action_scheme(self) -> ActionScheme:
        """
        The component for transforming an action into a `Trade` at each time step.
        获取交易动作方案，它根据每一个时间切片，将动作转换为交易指令
        """
        return self._action_scheme

    @action_scheme.setter
    def action_scheme(self, action_scheme: ActionScheme):
        """
        set action scheme
        设置交易动作策略实例
        :param action_scheme:
        :return:
        """
        self._action_scheme = action_scheme

    @property
    def reward_scheme(self) -> RewardScheme:
        """
        The component for determining the reward at each time step.
        获取奖赏方案，它决定每一步操作带来的奖赏
        """
        return self._reward_scheme

    @reward_scheme.setter
    def reward_scheme(self, reward_scheme: RewardScheme):
        """
        set reward scheme instance
        设置奖赏方案实例
        :param reward_scheme:
        :return:
        """
        self._reward_scheme = reward_scheme

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """
        The feature pipeline to pass the observations through.
        获取特征转换管道
        """
        return self._exchange.feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        """
        set feature pipline for transform
        设置特征转换管道
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
        executed_trade = self._action_scheme.get_trade(current_step=self._current_step,
                                                       action=action)
        filled_trade = self._exchange.execute_trade(executed_trade)
        return executed_trade, filled_trade

    def _next_observation(self) -> np.ndarray:
        """Returns the next observation from the exchange.
        获取下一步的观测值
        Returns:
            The observation provided by the environments's exchange, often OHLCV or tick trade history data points.
        """
        observation = self._exchange.next_observation()

        if len(observation) != 0:
            observation = observation[0]

        observation = np.nan_to_num(observation)

        return observation

    def _get_reward(self, trade: Trade) -> float:
        """Returns the reward for the current timestep.
        返回当前(步）切片的奖赏
        Returns:
            A float corresponding to the benefit earned by the action taken this step.
             根据当前动作交易产生的利润
        """
        reward = self._reward_scheme.get_reward(current_step=self._current_step,
                                                trade=trade)
        reward = np.nan_to_num(reward)

        if np.bitwise_not(np.isfinite(reward)):
            raise ValueError('Reward returned by the reward scheme must by a finite float.')

        return reward

    def _done(self) -> bool:
        """Returns whether or not the environments is done and should be restarted.
        如果环境没有下一步观测空间，或者净值过低，就返回终结标志
        Returns:
            A boolean signaling whether the environments is done and should be restarted.
        """
        lost_90_percent_net_worth = self._exchange.profit_loss_percent < 0.1
        return lost_90_percent_net_worth or not self._exchange.has_next_observation

    def _info(self, executed_trade: Trade, filled_trade: Trade, reward: int) -> dict:
        """Returns any auxiliary, diagnostic, or debugging information for the current timestep.
获取当前数据，包括当前步骤，交易账号，执行交易，完成的交易
        Returns:
            info: A dictionary containing the exchange used, the current timestep, and the filled trade, if any.
        """
        assert filled_trade.step == executed_trade.step

        return {
            'current_step': executed_trade.step,
            'executed_trade': executed_trade,
            'filled_trade': filled_trade,
            'reward': reward,
            'exchange': self._exchange,
        }

    def step(self, action) -> Tuple[pd.DataFrame, float, bool, dict]:
        """Run one timestep within the environments based on the specified action.
        根据指定的动作，执行一步。
        Arguments:
            action: The trade action provided by the agent for this timestep.

        Returns:
            observation (pandas.DataFrame): Provided by the environments's exchange, often OHLCV or tick trade history data points.
            reward (float): An amount corresponding to the benefit earned by the action taken this timestep.
            done (bool): If `True`, the environments is complete and should be restarted.
            info (dict): Any auxiliary, diagnostic, or debugging information to output.
        """
        executed_trade, filled_trade = self._take_action(action)

        observation = self._next_observation()
        reward = self._get_reward(filled_trade)
        done = self._done()
        info = self._info(executed_trade, filled_trade, reward)

        self._current_step += 1

        return observation, reward, done, info

    def reset(self) -> pd.DataFrame:
        """
        Resets the state of the environments and returns an initial observation.
        重置当前环境，返回初始化的观测空间
        Returns:
            The episode's initial observation.
        """
        # 重置交易动作方案
        self._action_scheme.reset()
        # 重置奖赏方案
        self._reward_scheme.reset()
        # 重置当前步骤记录数
        self._exchange.reset()

        self._current_step = 0

        observation = self._next_observation()

        self._current_step = 1

        return observation

    def render(self, mode='none'):
        """Renders the environment via matplotlib."""
        if mode == 'system':
            self.logger.info('Price: ' + str(self.exchange._current_price()))
            self.logger.info('Net worth: ' + str(self.exchange.performance[-1]['net_worth']))
        elif mode == 'human':
            if self.viewer is None:
                self.viewer = MatplotlibTradingChart(self.exchange.data_frame)

            self.viewer.render(self._current_step,
                               self.exchange.performance.loc[:, 'net_worth'],
                               self.render_benchmarks,
                               self.exchange.trades)

    def close(self):
        """Utility method to clean environment before closing."""
        if self.viewer is not None:
            self.viewer.close()

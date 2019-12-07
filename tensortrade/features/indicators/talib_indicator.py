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

import talib
import numpy as np
import pandas as pd

from gym import Space
from copy import copy
from abc import abstractmethod
from typing import Union, List, Callable

from tensortrade.features.feature_transformer import FeatureTransformer

class TAlibIndicator(FeatureTransformer):
    """
    Adds one or more TAlib indicators to a data frame, based on existing open, high, low, and close column values.
    添加一个或多个talib的指标，基于开/高/低/收字段
    """

    def __init__(self, indicators: List[str], lows: Union[List[float], List[int]] = None, highs: Union[List[float], List[int]] = None):
        # 指标列表
        self._indicator_names = indicators

        # 指标函数列表 str =》 talab.Method
        self._indicators = list(
            map(lambda indicator_name: self._str_to_indicator(indicator_name), indicators))

        # 指标最低值列表
        self._lows = lows or np.zeros(len(indicators))
        # 指标最高值列表
        self._highs = highs or np.ones(len(indicators))

    def _str_to_indicator(self, indicator_name: str):
        """
        根据指标名称，更换为Talib的方法属性
        """
        return getattr(talib, indicator_name.upper())

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        """
        转换空间维度
        :param input_space:
        :param column_names:
        :return:
        """
        # 复制输入空间 =》输出空间
        output_space = copy(input_space)

        shape_x, *shape_y = input_space.shape

        # 添加指标到x维度
        output_space.shape = (shape_x + len(self._indicators), *shape_y)

        # 逐一指标执行，
        for i in range(len(self._indicators)):
            # 输出空间中，该指标的最小值初始化
            output_space.low = np.append(output_space.low, self._lows[i])
            # 输出空间中，该指标的最大值初始化
            output_space.high = np.append(output_space.high, self._highs[i])

        return output_space

    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """
        输入pandas，输入空间，得出带有指标运算结果的输出空间
        :param X:
        :param input_space:
        :return:
        """
        for i in range(len(self._indicators)):
            indicator_name = self._indicator_names[i]
            indicator = self._indicators[i]

            X[indicator_name.upper()] = indicator(X['open'], X['high'], X['low'], X['close'])

        return X

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

from tensortrade.features import FeatureTransformer


class TAlibIndicator(FeatureTransformer):
    """
    Adds one or more TAlib indicators to a data frame, based on existing open, high, low, and close column values.
    添加一个或多个talib的指标，基于开/高/低/收字段
    """

    def __init__(self, indicators: List[str], lows: Union[List[float], List[int]] = None, highs: Union[List[float], List[int]] = None, **kwargs):
        # 指标列表
        self._indicator_names = [indicator[0].upper() for indicator in indicators]
        # 指标函数参数列表
        self._indicator_args = [indicator[1] for indicator in indicators]
        # 指标函数列表 str =》 talab.Method
        self._indicators = [getattr(talib, name) for name in self._indicator_names]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        特征转换
        :param X:
        :return:
        """
        # 逐一指标执行，
        for idx, indicator in enumerate(self._indicators):
            # 指标名称
            indicator_name = self._indicator_names[idx]
            # 指标参数
            indicator_args = [X[arg].values for arg in self._indicator_args[indicator_name]]

            # 特殊处理布林值（返回三个值）
            if indicator_name == 'BBANDS':
                upper, middle, lower = indicator(*indicator_args)

                X["bb_upper"] = upper
                X["bb_middle"] = middle
                X["bb_lower"] = lower
            else:
                # 普通指标，只有一个返回值
                try:
                    value = indicator(*indicator_args)

                    if type(value) == tuple:
                        X[indicator_name] = value[0][0]
                    else:
                        X[indicator_name] = value

                except:
                    X[indicator_name] = indicator(*indicator_args)[0]

        return X

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

from gym import Space
from copy import copy
from typing import Union, List, Tuple

from tensortrade.features.feature_transformer import FeatureTransformer


class StandardNormalizer(FeatureTransformer):
    """
    A transformer for normalizing values within a feature pipeline by removing the mean and scaling to unit variance.
    标准化常态转化器
    """

    def __init__(self, columns: Union[List[str], str, None] = None, feature_min=0, feature_max=1, inplace=True):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            feature_min (optional): The minimum value in the range to scale to.
            feature_max (optional): The maximum value in the range to scale to.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        # 特征最小值
        self._feature_min = feature_min
        # 特征最大值
        self._feature_max = feature_max
        # 更换数据
        self._inplace = inplace
        # 转换的字段
        self.columns = columns

        # 历史数据
        self._history = {}

    def reset(self):
        self._history = {}

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        """
        转换空间
        :param input_space:
        :param column_names:
        :return:
        """
        # 如果已存在输入空间，直接返回
        if self._inplace:
            return input_space
        # 复制
        output_space = copy(input_space)
        # 获取 x，y 维度大小
        shape_x, *shape_y = input_space.shape
        # 当前字段列表
        columns = self.columns or range(len(shape_x))
        # 输出空间的share，扩展了字段数量
        output_space.shape = (shape_x + len(columns), *shape_y)

        for _ in columns:
            # 填充缺省最小值
            output_space.low = np.append(output_space.low, self._feature_min)
            # 填充缺省最高值
            output_space.high = np.append(output_space.high, self._feature_max)

        return output_space

    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """

        :param X:
        :param input_space:
        :return:
        """
        if self.columns is None:
            self.columns = list(X.columns)

        raise NotImplementedError

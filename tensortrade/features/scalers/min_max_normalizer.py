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
from typing import Union, List, Tuple, Dict

from tensortrade.features.feature_transformer import FeatureTransformer


class MinMaxNormalizer(FeatureTransformer):
    """
    A transformer for normalizing values within a feature pipeline by the column-wise extrema.
    特征转换器，按照最小/最大值进行数据常态化
    根据指定的columns列表，给出的最大最小值范围。
    """

    def __init__(self,
                 columns: Union[List[str], str, None] = None,
                 input_min: float = -1E-8,
                 input_max: float = 1E8,
                 feature_min: float = 0,
                 feature_max: float = 1,
                 inplace: bool = True):
        """
        Arguments:
            columns (optional): A list of column names to normalize.
            input_min (optional): The minimum `float` in the range to scale to. Defaults to -1E-8.
            input_max (optional): The maximum `float` in the range to scale to. Defaults to 1E8.
            feature_min (optional): The minimum `float` in the range to scale to. Defaults to 0.
            feature_max (optional): The maximum `float` in the range to scale to. Defaults to 1.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        super().__init__(columns=columns, inplace=inplace)
        # 输入的最小值
        self._input_min = input_min
        # 输入的最大值
        self._input_max = input_max
        # 特征值最小值
        self._feature_min = feature_min
        # 特征值最大值
        self._feature_max = feature_max

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
         特征转换
        :param X: 输入矩阵
        :return: 输出矩阵
        """
        # 如果当前columns为空，使用输入的df的columns替代
        if self.columns is None:
            self.columns = list(X.select_dtypes('number').columns)

        for column in self.columns:
            # 约束了最低值，最高值
            low, high = self._input_min, self._input_max
            # 指标范围
            scale = (self._feature_max - self._feature_min) + self._feature_min

            if high - low == 0:
                # 最高和最低一致：一条直线
                normalized_column = (1/len(X[column])) * scale
            else:
                # 对指标进行指标化计算
                normalized_column = (X[column] - low) / (high - low) * scale

            if not self._inplace:
                # 另起新的名称保存常态化之后的值
                column = '{}_minmax_{}_{}'.format(column, self._feature_min, self._feature_max)

            args = {}
            args[column] = normalized_column

            X = X.assign(**args)

        return X

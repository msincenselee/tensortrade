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
#
# Reference Source: Marcos Lopez De Prado - Advances in Financial Machine Learning
#                   Chapter 5 (Pg. 82) - Fractionally Differentiated Features

import pandas as pd
import numpy as np

from gym import Space
from copy import copy
from typing import Union, List, Tuple

from tensortrade.features.feature_transformer import FeatureTransformer


class FractionalDifference(FeatureTransformer):
    """
    A transformer for differencing values within a feature pipeline by a fractional order.
    分数阶差转换器
    """

    def __init__(self,
                 columns: Union[List[str], str, None] = None,
                 difference_order: float = 0.5,
                 difference_threshold: float = 1e-1,
                 inplace: bool = True):
        """
        Arguments:
            columns (optional): A list of column names to difference.
            difference_order (optional): The fractional difference order. Defaults to 0.5.
            inplace (optional): If `False`, a new column will be added to the output for each input column.
        """
        # 需要差分处理的字段
        self.columns = columns

        # 阶差顺序
        self._difference_order = difference_order
        # 阶差基准
        self._difference_threshold = difference_threshold

        # 替换更新
        self._inplace = inplace

        self.reset()

    def reset(self):
        """重置"""
        self._history = None

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        """
        转换空间
        :param input_space:
        :param column_names:
        :return:
        """
        # 如果已经定义好转换空间，直接使用
        if self._inplace:
            return input_space
        # 复制
        output_space = copy(input_space)
        # 字段
        columns = self.columns or column_names
        # 输入空间的x，y
        shape_x, *shape_y = input_space.shape
        # 扩展x数量
        output_space.shape = (shape_x + len(columns), *shape_y)

        for column in columns:
            # 字段所在下标索引
            column_index = column_names.index(column)
            # 获取该字段的最低/最高值
            low, high = input_space.low[column_index], input_space.high[column_index]
            # 初始化最低值为：最低-最高
            output_space.low = np.append(output_space.low - output_space.high, low)
            # 初始化最高值
            output_space.high = np.append(output_space.high, high)

        return output_space

    def _difference_weights(self, size: int):
        """生成阶差权重"""
        weights = [1.0]

        for k in range(1, size):
            weight = -weights[-1] / k * (self._difference_order - k + 1)
            weights.append(weight)

        return np.array(weights[::-1]).reshape(-1, 1)

    def _fractional_difference(self, series: pd.Series):
        """Computes fractionally differenced series, with an increasing window width.
        计算分数阶差序列，
        Args:
            series: A `pandas.Series` to difference by self._difference_order with self._difference_threshold.

        Returns:
            The fractionally differenced series.
        """
        # 阶差权重
        weights = self._difference_weights(len(series))

        # 按行累加
        weight_sums = np.cumsum(abs(weights))
        # 累加变化率
        weight_sums /= weight_sums[-1]
        # 忽略的权重。大于基准值的数量
        skip_weights = len(weight_sums[weight_sums > self._difference_threshold])
        # 当前序列
        curr_series = series.dropna()
        # 创建新的阶差序列
        diff_series = pd.Series(index=series.index)

        for current_index in range(skip_weights, curr_series.shape[0]):
            index = curr_series.index[current_index]

            if not np.isfinite(curr_series.loc[index]):
                continue

            # 得出阶差值
            diff_series[index] = np.dot(
                weights[-(current_index + 1):, :].T, curr_series.loc[:index])[0]

        return diff_series

    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """
        特征转换
        :param X:
        :param input_space:
        :return:
        """
        # 记录历史
        if self._history is None:
            # 复制
            self._history = X.copy()
        else:
            # 向dataframe对象中添加新的行，如果添加的列名不在dataframe对象中，将会被当作新的列进行添加
            # 不使用索引
            self._history = self._history.append(X, ignore_index=True)

        # 行数超过X的行数，裁剪最后跟X长度一致
        if len(self._history) > len(X):
            self._history = self._history.iloc[-len(X) + 1:]

        # 如果columns为空，使用X的columns
        if self.columns is None:
            self.columns = list(X.columns)

        for column in self.columns:
            # 通过对历史数据进行阶差，得出新的阶差序列
            diffed_series = self._fractional_difference(self._history[column])

            if self._inplace:
                # 更新值
                X[column] = diffed_series.fillna(method='bfill')
            else:
                # 添加新的阶差字段
                column_name = '{}_diff_{}'.format(column, self._difference_order)
                X[column_name] = diffed_series.fillna(method='bfill')

        return X.iloc[-len(X):]

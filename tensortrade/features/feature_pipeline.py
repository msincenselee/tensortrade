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

import pandas as pd
import numpy as np

from gym import Space
from typing import List, Union, Callable

from .feature_transformer import FeatureTransformer

DTypeString = Union[type, str]


class FeaturePipeline(object):
    """
    An pipeline for transforming observation data frames into features for learning.
    特征向量库，用于把观测数据=》各类特征向量，用作机器学习
    """

    def __init__(self, steps: List[FeatureTransformer], **kwargs):
        """
        Arguments:
            dtype: The `dtype` elements in the pipeline should be cast to.
        """
        # 特征计算器列表
        self._steps = steps

        self._dtype: DTypeString = kwargs.get('dtype', np.float16)

    @property
    def steps(self) -> List[FeatureTransformer]:
        """
        A list of feature transformations to apply to observations.
        获取实现观测值特征计算的转换器列表
        """
        return self._steps

    @steps.setter
    def steps(self, steps: List[FeatureTransformer]):
        self._steps = steps

    @property
    def dtype(self) -> DTypeString:
        """
        The `dtype` that elements in the pipeline should be input and output as.
        获取特征库类型
        """
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        """设置特征库"""
        self._dtype = dtype

    def reset(self):
        """
        Reset all transformers within the feature pipeline.
        重置所有的转换器
        """
        for transformer in self._steps:
            transformer.reset()

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        """Get the transformed output space for a given input space.
        输入观测空间和对应数据项，返回转换空间
        Args:
            input_space: A `gym.Space` matching the shape of the pipeline's input.
            column_names: A list of all column names in the input data frame.

        Returns:
            A `gym.Space` matching the shape of the pipeline's output.
        """
        # 定义输出空间
        output_space = input_space

        # 根据转换器，逐一更新输出空间
        for transformer in self._steps:
            output_space = transformer.transform_space(output_space, column_names)

        return output_space

    def _transform(self, observations: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """
        Utility method for transforming observations via a list of `FeatureTransformer` objects.
        工具方法，对所有观测数据，进行所有转换器执行，返回观测矩阵
        """
        # 逐一执行转换器
        for transformer in self._steps:
            observations = transformer.transform(observations, input_space)

        return observations

    def transform(self, observation: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """Apply the pipeline of feature transformations to an observation frame.
        使用特征转换器，对观测数据转换，得到观测矩阵
        Arguments:
            observation: A `pandas.DataFrame` corresponding to an observation within a `TradingEnvironment`.
            input_space: A `gym.Space` matching the shape of the pipeline's input.

        Returns:
            A `pandas.DataFrame` of features corresponding to an input oversvation.

        Raises:
            ValueError: In the case that an invalid observation frame has been input.
        """
        features = self._transform(observation, input_space)

        if not isinstance(features, pd.DataFrame):
            raise ValueError("A FeaturePipeline must transform a pandas.DataFrame into another pandas.DataFrame.\n \
                               Expected return type: {} `\n \
                               Actual return type: {}.".format(type(pd.DataFrame([])), type(features)))

        return features

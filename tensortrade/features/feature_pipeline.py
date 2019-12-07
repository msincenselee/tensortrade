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

from tensortrade import Component
from .feature_transformer import FeatureTransformer


DTypeString = Union[type, str]


class FeaturePipeline(Component):
    """
    An pipeline for transforming observation data frames into features for learning.
    特征向量管道，用于把观测数据=》各类特征向量，用作机器学习

    """
    registered_name = "features"

    def __init__(self, steps: List[FeatureTransformer], **kwargs):
        """
        Arguments:
            dtype: The `dtype` elements in the pipeline should be cast to.
        """
        # 特征计算器列表
        self._steps = steps

        self._dtype: DTypeString = self.default('dtype', np.float32, kwargs)

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

    def _transform(self, observations: pd.DataFrame) -> pd.DataFrame:
        """
        Utility method for transforming observations via a list of `FeatureTransformer` objects.
        工具方法，对所有观测数据，进行所有转换器执行，返回观测矩阵
        """
        for transformer in self._steps:
            observations = transformer.transform(observations)

        return observations

    def transform(self, observation: pd.DataFrame) -> pd.DataFrame:
        """Apply the pipeline of feature transformations to an observation frame.
        使用特征转换器，对观测数据转换，得到观测矩阵
        Arguments:
            observation: A `pandas.DataFrame` corresponding to an observation within a `TradingEnvironment`.

        Returns:
            A `pandas.DataFrame` of features corresponding to an input oversvation.

        Raises:
            ValueError: In the case that an invalid observation frame has been input.
        """
        obs = observation.copy(deep=True)
        features = self._transform(obs)

        if not isinstance(features, pd.DataFrame):
            raise ValueError("A FeaturePipeline must transform a pandas.DataFrame into another pandas.DataFrame.\n \
                               Expected return type: {} `\n \
                               Actual return type: {}.".format(type(pd.DataFrame([])), type(features)))

        return features

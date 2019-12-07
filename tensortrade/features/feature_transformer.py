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

from gym import Space
from typing import List, Union
from abc import ABCMeta, abstractmethod


class FeatureTransformer(object, metaclass=ABCMeta):
    """
    An abstract feature transformer for use within feature pipelines.
    抽象类，特征转换器，
    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def columns(self) -> List[str]:
        """获取字段列表"""
        return self._columns

    @columns.setter
    def columns(self, columns=Union[List[str], str]):
        """设置字段列表"""
        self._columns = columns

        # 单字符串 =》 列表
        if isinstance(self._columns, str):
            self._columns = [self._columns]

    def reset(self):
        """
        Optionally implementable method for resetting stateful transformers.
        可选方法，重置带有状态的转换器
        """
        pass

    @abstractmethod
    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        """Get the transformed output space for a given input space.
        抽象方法，需要实现：根据输入空间维度和指定的数据项列表，返回输出空间维度
        Args:
            input_space: A `gym.Space` matching the shape of the pipeline's input.
                         符合 gym.Space 规则的输入空间维度
            column_names: A list of all column names in the input data frame.

        Returns:
            A `gym.Space` matching the shape of the pipeline's output.
            符合gym.space规则的输出空间维度
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """Transform the data set and return a new data frame.
        抽象方法，需要实现：根据数据和观测空间维度，转换为输出空间
        Arguments:
            X: The set of data to transform.
            input_space: A `gym.Space` matching the shape of the pipeline's input.

        Returns:
            A transformed data frame.
        """
        raise NotImplementedError

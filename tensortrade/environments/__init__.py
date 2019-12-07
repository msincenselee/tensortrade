from .trading_environment import TradingEnvironment

from . import render

_registry = {
    # 基本配置
    'basic': {
        'exchange': 'simulated',     # 交易账号，模拟
        'action_scheme': 'discrete', # 动作方案： 离散
        'reward_scheme': 'simple'    # 奖励方案： 简单
    }
}


def get(identifier: str) -> TradingEnvironment:
    """Gets the `TradingEnvironment` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `TradingEnvironment`

    Raises:
        KeyError: if identifier is not associated with any `TradingEnvironment`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `TradingEnvironment`.'.format(identifier))
    return TradingEnvironment(**_registry[identifier])

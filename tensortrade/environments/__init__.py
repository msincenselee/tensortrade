from .trading_environment import TradingEnvironment

# 环境字典
_registry = {
    # 基本配置
    'basic': {
        'exchange': 'simulated',        # 交易账号，模拟
        'action_strategy': 'discrete',  # 交易策略： 离散
        'reward_strategy': 'simple'     # 奖励策略： 简单
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

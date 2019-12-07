from .action_strategy import ActionStrategy, DTypeString, TradeActionUnion
from .continuous_action_strategy import ContinuousActionStrategy
from .discrete_action_strategy import DiscreteActionStrategy
from .multi_discrete_action_strategy import MultiDiscreteActionStrategy


# 交易动作策略字典
_registry = {
    'continuous': ContinuousActionStrategy(),
    'discrete': DiscreteActionStrategy(),
    'multi-discrete': MultiDiscreteActionStrategy(instrument_symbols=['BTC', 'ETH']),
}


def get(identifier: str) -> ActionStrategy:
    """Gets the `ActionStrategy` that matches with the identifier.
       通过identifier标识，获取相应的动作策略

    Arguments:
        identifier: The identifier for the `ActionStrategy` 策略标识

    Raises:
        KeyError: if identifier is not associated with any `ActionStrategy` 不存在该标识
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `ActionStrategy`.'.format(identifier))
    return _registry[identifier]

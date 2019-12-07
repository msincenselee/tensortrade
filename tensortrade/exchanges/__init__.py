import ccxt
import pandas as pd

from datetime import datetime

from .exchange import Exchange

from . import live
from . import simulated

# 交易账号字典
_registry = {
    'simulated': simulated.SimulatedExchange,
    'fbm': simulated.FBMExchange,
    'gan': simulated.GANExchange
}


def get(identifier: str) -> Exchange:
    """Gets the `Exchange` that matches with the identifier.
    根据标识符获取交易账号实例
    As a caution, when exchanges that require a data frame are instantiated by
    this function, the data frame is set as None and must be set at a later
    point in time for the exchange to work.

    Arguments:
        identifier: The identifier for the `Exchange`

    Raises:
        KeyError: if identifier is not associated with any `Exchange`
    """
    if identifier in _registry.keys():
        if identifier == 'simulated':
            data_frame = pd.DataFrame(
                [{'open': 1, 'high': 1.5, 'low': 0.5, 'close': 1.1, 'volume': 1000}])
            return _registry['simulated'](data_frame=data_frame)

        return _registry[identifier]()

    if identifier in ccxt.exchanges:
        return live.CCXTExchange(exchange=identifier)

    raise KeyError('Identifier {} is not associated with any `Exchange`.'.format(identifier))

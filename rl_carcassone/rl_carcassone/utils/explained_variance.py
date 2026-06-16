from typing import Union

import numpy as np
import torch


def explained_variance(values: Union[torch.Tensor, np.ndarray], returns: Union[torch.Tensor, np.ndarray]) -> float:
    if isinstance(values, torch.Tensor):
        var_method = torch.var
    else:
        var_method = np.var

    var_returns = var_method(returns).item()
    if var_returns == 0:
        return np.nan
    else:
        return 1 - var_method(returns - values).item() / var_returns

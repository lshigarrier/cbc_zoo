import torch
from .adp_souple import ADPSouple
from .wsss import WSSS


def load(model_name: str, verbose: bool = False) -> torch.nn.Module:
    if model_name == 'ADPSouple':
        return ADPSouple(verbose)
    elif model_name == 'WSSS':
        return WSSS(verbose)
    else:
        raise NotImplementedError

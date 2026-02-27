import torch
from .adp_souple import ADPSouple


def get_cbc_model(model_name: str, device: str | torch.device = 'cpu', verbose: bool = False) -> ADPSouple:
    if model_name == 'ADPSouple':
        return ADPSouple(device, verbose)
    else:
        raise NotImplementedError

from .adp_souple import ADPSouple


def get_cbc_model(model_name, device='cpu', verbose=False):
    if model_name == 'ADPSouple':
        return ADPSouple(device, verbose)
    else:
        raise NotImplementedError

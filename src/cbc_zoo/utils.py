import logging


def print_parameters(model):
    logging.info('-'*70)
    for name, par in model.named_parameters():
        logging.info(f'{name}, {par.shape}')
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('-'*70)
    logging.info(f'Trainable parameters: {num_param}')

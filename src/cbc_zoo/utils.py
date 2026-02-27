import torch.nn as nn
import logging

def log_parameters(model: nn.Module, logger: logging.Logger) -> None:
    logger.info('-'*70)
    for name, par in model.named_parameters():
        logger.info(f'{name}, {par.shape}')
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('-'*70)
    logger.info(f'Trainable parameters: {num_param}')

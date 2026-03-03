import torch
import torchvision.transforms.v2 as transforms
import logging
from pathlib import Path

from .utils import log_parameters


class ADPSouple(torch.nn.Module):

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.logger = logging.getLogger('ADPSouple')
        self.transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ])
        base_dir = Path(__file__).parent
        model_path = base_dir / 'models' / 'my_model_unet_aspp_adam_CE_45_weight_1_e_5_learning_rate_v2.pth'
        self.model = torch.jit.load(model_path)
        self.model.eval()
        if verbose:
            log_parameters(self.model, self.logger)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            image = self.transform(image)
            # Add a batch dimension if necessary (models expect [B, C, H, W])
            if image.ndim == 3:
                image = image.unsqueeze(0)
            return self.model(image)  # output shape : B x 3 x H x W

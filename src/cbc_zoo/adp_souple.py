import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
import logging
import os
import numpy as np
from pathlib import Path
from PIL import Image

from .utils import print_parameters


def mask_to_rgba(mask):
    """
    mask : (H, W) numpy array avec valeurs 0, 1, 2
    Retourne : (H, W, 4) numpy array uint8 RGBA
    """
    cmap = np.array([
        [0, 0, 0, 0],  # 0 → noir (ou transparent)
        [255, 0, 0, 1],  # 1 → rouge
        [0, 0, 255, 1],  # 2 → bleu
    ], dtype=np.uint8)
    return cmap[mask]


class ADPSouple:

    def __init__(self, device='cpu', verbose=False):
        self.device = device
        self.verbose = verbose
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
             transforms.ToDtype(torch.float32, scale=True),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ])
        base_dir = Path(__file__).parent
        model_path = base_dir / 'models' / 'my_model_unet_aspp_adam_CE_45_weight_1_e_5_learning_rate_v2.pth'
        self.model = torch.jit.load(model_path)
        self.model.to(device)
        self.model.eval()
        if verbose:
            logging.info('=' * 70)
            logging.info('ADP Souple')
            logging.info(f'device: {device}')
            print_parameters(self.model)

    def to(self, device):
        self.device = device
        self.model.to(device)

    def __call__(self, image, output_dir=None, image_name=None):
        image_trans = self.transform(image)
        image_trans = image_trans.to(self.device)
        # Add a batch dimension if necessary (models expect [B, C, H, W])
        if image_trans.ndim == 3:
            image_trans = image_trans.unsqueeze(0)

        output = self.model(image_trans)  # output shape : B x 3 x 512 x 512

        if output_dir is not None and image_name is not None:
            batch_size, _, img_h, img_w = image.shape
            output_trans = F.interpolate(
                output,
                size=(img_h, img_w),
                mode="bicubic",
                align_corners=False
            )
            mask = torch.argmax(output_trans, dim=1).cpu().numpy()

            for i in range(batch_size):
                rgba = mask_to_rgba(mask[i])
                os.makedirs(output_dir, exist_ok=True)
                Image.fromarray(rgba[..., :3]).save(output_dir / f'{image_name}_mask.png')
                mask_a = rgba[..., 3:].astype(float)
                blended = image[i].astype(float) * (1 - mask_a * 0.5) + rgba[..., :3].astype(float) * (mask_a * 0.5)
                blended = blended.clip(0, 255).astype(np.uint8)

                Image.fromarray(blended).save(output_dir / f'{image_name}_over.png')

        return output

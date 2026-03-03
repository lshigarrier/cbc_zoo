import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import logging
from pathlib import Path
from typing import Any

from .utils import log_parameters


class WSSS(torch.nn.Module):

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.logger = logging.getLogger('WSSS')
        self.transform = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=False),
            transforms.Normalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
        ])
        base_dir = Path(__file__).parent
        model_path = base_dir / 'models' / 'CoSA_LCC_5.pt'
        self.model = torch.jit.load(model_path)
        self.model.eval()
        if verbose:
            log_parameters(self.model, self.logger)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any]:
        with torch.inference_mode():
            image = self.transform(image)
            # Add a batch dimension if necessary
            if image.ndim == 3:
                image = image.unsqueeze(0)
            b, c, h, w = image.shape
            cam_list = []
            cam_aux_list = []
            cls_final = 0
            scales = [1.0, 0.5, 1.5, 0.75, 1.25]
            for s in scales:
                if s != 1.0:
                    imgs = F.interpolate(image, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                else:
                    imgs = image
                imgs_cat = torch.cat([imgs, imgs.flip(-1)], dim=0)
                cls_f, cls_a, _, _, _cam, _cam_aux = self.model(imgs_cat)
                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

                cls_final += torch.sum(cls_f, dim=0, keepdim=True)

            cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
            cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
            cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

            cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
            cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
            cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

            return cam, cam_aux, cls_final

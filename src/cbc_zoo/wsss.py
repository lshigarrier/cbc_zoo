import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import logging
from pathlib import Path

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
        model_path = base_dir / 'models' / 'CoSA_LCC_7_final.pt'
        self.invert_activation = True  # set to True if using CoSA_LCC_7, set to False if using CoSA_LCC_5
        self.model = torch.jit.load(model_path)
        self.model.eval()
        if verbose:
            log_parameters(self.model, self.logger)

    def forward(self, image: torch.Tensor) -> tuple:
        with torch.inference_mode():
            image = self.transform(image)
            # Add a batch dimension if necessary
            if image.ndim == 3:
                image = image.unsqueeze(0)

            # multi_scale_camsegv3
            b, c, h, w = image.shape
            cam_list = []
            cam_aux_list = []
            logits_final = 0
            scales = [1.0, 0.5, 1.5, 0.75, 1.25]
            for s in scales:
                if s != 1.0:
                    imgs = F.interpolate(image, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                else:
                    imgs = image
                imgs_cat = torch.cat([imgs, imgs.flip(-1)], dim=0)
                cls_f, _, _, _, _cam, _cam_aux = self.model(imgs_cat)
                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

                logits_final += cls_f[:b]
                logits_final += cls_f[b:]

            cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
            cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
            cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

            cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
            cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
            cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

            # CAM validation
            mix_cam_avg = (cam + cam_aux) / 2
            if self.invert_activation:
                # invert activation for RES
                mix_cam_avg[:, 1] = 1 - mix_cam_avg[:, 1]
                # invert activation for COS
                mix_cam_avg[:, 3] = 1 - mix_cam_avg[:, 3]
            cls_label_rep = logits_final.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
            valid_cam = cls_label_rep * mix_cam_avg

            return mix_cam_avg, valid_cam, logits_final

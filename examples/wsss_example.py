import logging
import os
import torch
import matplotlib
import numpy as np
import sklearn.mixture as skm
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

from utils_examples import get_device, CustomTimer, log_memory

import cbc_zoo as cbc

logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(message)s')


def mask_to_rgba(mask):
    """
    Args:
        mask: (..., H, W) numpy array with integer values 0, ..., 6

    Returns:
        (..., H, W, 4) numpy array uint8 RGBA
    """
    cmap = np.array([
        [0, 0, 0, 0],  # 0 (0) → black or transparent (pas de dégradation)
        [255, 0, 0, 1],  # 1 (1) → red (fissure souple)
        [0, 255, 0, 1],  # 2 (5) → green (dépôt de gomme)
        [0, 255, 255, 1],  # 3 (7) → cyan (arrachement rigide)
        [255, 255, 0, 1],  # 4 (8) → yellow (réparation rigide)
        [255, 0, 255, 1],  # 5 (10) → magenta (fissure rigide)
        [0, 0, 255, 1],  # 6 (12) → blue (joint de dalle)
    ], dtype=np.uint8)
    return cmap[mask]


def index_to_label(idx):
    labels = [
        'FIS',
        'GOM',
        'ARR',
        'RER',
        'FIR',
        'JOI',
    ]
    return labels[idx]


def rungmm_eval(cam, percentile=75, filter_thre=0.05):
    flat_cam = cam.flatten()
    flat_cam = flat_cam[flat_cam > filter_thre]
    if len(flat_cam) == 0:
        return 1.
    flat_cam = flat_cam.reshape(-1, 1)
    means_init = [[np.min(flat_cam)], [np.max(flat_cam)]]
    weights_init = [1/2, 1/2]
    precisions_init = [[[1.0]], [[1.0]]]
    gmm = skm.GaussianMixture(2,
                              weights_init=weights_init, means_init=means_init, precisions_init=precisions_init)
    prediction = gmm.fit_predict(flat_cam)
    group0 = flat_cam[prediction == 0]
    group1 = flat_cam[prediction == 1]
    if len(group1) > 0:
        return np.percentile(group1, percentile).item()
    elif len(group0) > 0:
        # return np.max(group0).item()
        return np.percentile(group0, percentile).item()
    else:
        raise RuntimeError


def process_and_save(images, names, outputs, boxes, output_dir, dataset):
    cmap = matplotlib.colormaps['hot']
    os.makedirs(output_dir, exist_ok=True)

    mix_cam, valid_cam, logits = outputs
    valid_cam = dataset.stitch(valid_cam, boxes)  # list((6, H, W))
    mix_cam = dataset.stitch(mix_cam, boxes)  # list((6, H, W))
    batch_size = len(valid_cam)
    logits = logits.reshape(batch_size, dataset.patch_per_img, -1).amax(dim=1)  # (batch_size, 6)

    for b in range(batch_size):
        # CAM to label
        cam_value, mask = valid_cam[b].max(dim=0, keepdim=False)
        mask += 1
        bkg_thre = rungmm_eval(cam_value.cpu().numpy())
        mask[cam_value <= bkg_thre] = 0

        # Save mask
        mask = mask_to_rgba(mask.cpu().numpy())
        mask_a = mask[:, :, 3:].astype(np.float32)
        mask = mask[:, :, :3].astype(np.float32)
        image = images[b].numpy().transpose(1, 2, 0).astype(np.float32) * (1 - mask_a * 0.5) + mask * (mask_a * 0.5)
        image = image.clip(0, 255)
        Image.fromarray(mask.astype(np.uint8)).save(output_dir / f'{names[b]}_mask.png')
        Image.fromarray(image.astype(np.uint8)).save(output_dir / f'{names[b]}_over.png')

        # Save CAM
        for i, logit in enumerate(logits[b]):
            if logit.item() > 0:
                cam_rgb = cmap(mix_cam[b][i].cpu().numpy())[:, :, :3] * 255
                Image.fromarray(cam_rgb.astype(np.uint8)).save(output_dir / f'{names[b]}_CAM_{index_to_label(i)}.png')


def main():
    device = get_device()
    wsss = cbc.load('WSSS', verbose=True)
    wsss = wsss.to(device)
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'example_outputs' / 'WSSS'
    # num_patches_with_overlap = ceil((num_patches_without_overlap - 1) / (1 - patch_overlap)) + 1
    dataset = cbc.ImageDataset(
        base_dir / 'example_images',
        patch_per_row=8,
        patch_per_col=4,
        patch_size=448,
        patch_overlap=0.1
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=cbc.image_collate_fn
    )

    logger = logging.getLogger('WSSSExample')
    logger.info('-' * 70)
    logger.info(f'WSSS inference on {len(dataset)} example images')
    custom_timer = CustomTimer()
    custom_timer.start()

    with torch.inference_mode():
        for images, names, patches, boxes in dataloader:
            patches = patches.to(device)
            outputs = wsss(patches)
            process_and_save(images, names, outputs, boxes, output_dir, dataset)

    custom_timer.stop(logger, len(dataset))
    log_memory(logger)
    logger.info(f'Predictions saved in {output_dir}')


if __name__ == '__main__':
    main()

import logging
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

from utils_examples import get_device, CustomTimer, log_memory

import cbc_zoo as cbc

logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(message)s')


def mask_to_rgba(mask):
    """
    Args:
        mask: (..., H, W) numpy array with integer values 0, 1, 2

    Returns:
        (..., H, W, 4) numpy array uint8 RGBA
    """
    cmap = np.array([
        [0, 0, 0, 0],  # 0 → black or transparent (no defect)
        [255, 0, 0, 1],  # 1 → red (crack)
        [0, 0, 255, 1],  # 2 → blue (bridging)
    ], dtype=np.uint8)
    return cmap[mask]


def process_and_save(images, names, outputs, boxes, output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)

    logits = dataset.stitch(outputs, boxes)  # list((3, H, W))

    for b in range(len(images)):
        mask = torch.argmax(logits[b], dim=0).cpu().numpy()
        mask = mask_to_rgba(mask)
        mask_a = mask[:, :, 3:].astype(np.float32)
        mask = mask[:, :, :3].astype(np.float32)
        image = images[b].numpy().transpose(1, 2, 0).astype(np.float32) * (1 - mask_a * 0.5) + mask * (mask_a * 0.5)
        image = image.clip(0, 255)
        Image.fromarray(mask.astype(np.uint8)).save(output_dir / f'{names[b]}_mask.png')
        Image.fromarray(image.astype(np.uint8)).save(output_dir / f'{names[b]}_over.png')


def main():
    device = get_device()
    adpsouple = cbc.load('ADPSouple', verbose=True)
    adpsouple = adpsouple.to(device)
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'example_outputs' / 'ADPSouple'
    # num_patches_with_overlap = ceil((num_patches_without_overlap - 1) / (1 - patch_overlap)) + 1
    dataset = cbc.ImageDataset(
        base_dir / 'example_images',
        patch_per_row=8,
        patch_per_col=4,
        patch_size=512,
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

    logger = logging.getLogger('ADPSoupleExample')
    logger.info('-' * 70)
    logger.info(f'ADP Souple inference on {len(dataset)} example images')
    custom_timer = CustomTimer()
    custom_timer.start()

    with torch.inference_mode():
        for images, names, patches, boxes in dataloader:
            patches = patches.to(device)
            outputs = adpsouple(patches)
            process_and_save(images, names, outputs, boxes, output_dir, dataset)

    custom_timer.stop(logger, len(dataset))
    log_memory(logger)
    logger.info(f'Predictions saved in {output_dir}')


if __name__ == '__main__':
    main()

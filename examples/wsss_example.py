import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from utils_examples import get_device, CustomTimer, log_memory

import cbc_zoo as cbc

logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(message)s')


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
            # process_and_save(images, names, outputs, boxes, output_dir, dataset)

    custom_timer.stop(logger, len(dataset))
    log_memory(logger)
    logger.info(f'Predictions saved in {output_dir}')


if __name__ == '__main__':
    main()
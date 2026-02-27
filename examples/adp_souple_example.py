import logging
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from pathlib import Path
from PIL import Image

from utils_examples import get_device, CustomTimer, log_memory

import cbc_zoo as cbc

logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(message)s')

PATCH_PER_IMG = 32
PATCH_PER_ROW = 8
PATCH_PER_COL = 4
PATCH_SIZE = 512


class ImageDataset(Dataset):

    def __init__(self, folder):
        super().__init__()
        self.paths = sorted(Path(folder).glob('*.jpg'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        img = read_image(str(path))
        name = path.stem

        rows = img.chunk(PATCH_PER_COL, dim=1)
        patches = [p for row in rows for p in row.chunk(PATCH_PER_ROW, dim=2)]
        sizes = tuple((p.shape[1], p.shape[2]) for p in patches)
        patches = torch.stack(patches)
        patches = F.interpolate(patches, size=(PATCH_SIZE, PATCH_SIZE), mode='bicubic', align_corners=False)

        return img, name, patches, sizes


def custom_collate_fn(batch):
    all_imgs = tuple(item[0] for item in batch)
    all_names = tuple(item[1] for item in batch)
    all_patches = torch.cat([item[2] for item in batch], dim=0)
    all_sizes = sum((item[3] for item in batch), ())
    return all_imgs, all_names, all_patches, all_sizes


def mask_to_rgba(mask):
    """
    Args:
        mask: (H, W) numpy array with integer values 0, 1, 2

    Returns:
        (H, W, 4) numpy array uint8 RGBA
    """
    cmap = np.array([
        [0, 0, 0, 0],  # 0 → black or transparent (no defect)
        [255, 0, 0, 1],  # 1 → red (crack)
        [0, 0, 255, 1],  # 2 → blue (bridging)
    ], dtype=np.uint8)
    return cmap[mask]


def process_and_save(images, names, outputs, sizes, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    resized_patches = [
        F.interpolate(outputs[i:i+1], size=sizes[i], mode='bicubic', align_corners=False)
        for i in range(outputs.shape[0])
    ]
    batch_size = len(images)
    for b in range(batch_size):
        current_patches = resized_patches[b * PATCH_PER_IMG: (b + 1) * PATCH_PER_IMG]
        cat_rows = []
        for i in range(PATCH_PER_COL):
            cat_rows.append(torch.cat(current_patches[i * PATCH_PER_ROW: (i + 1) * PATCH_PER_ROW], dim=3))
        full_output = torch.cat(cat_rows, dim=2)[0]
        # (num_classes, H, W)

        mask = torch.argmax(full_output, dim=0).cpu().numpy()
        mask = mask_to_rgba(mask)
        mask_a = mask[..., 3:].astype(np.float32)
        image = (images[b].numpy().transpose(1, 2, 0).astype(np.float32) * (1 - mask_a * 0.5)
                 + mask[..., :3].astype(np.float32) * (mask_a * 0.5))
        image = image.clip(0, 255).astype(np.uint8)

        Image.fromarray(mask[:, :, :3]).save(output_dir / f'{names[b]}_mask.png')
        Image.fromarray(image).save(output_dir / f'{names[b]}_over.png')


def main():
    device = get_device()
    adpsouple = cbc.get_cbc_model('ADPSouple', device, verbose=True)
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'example_outputs' / 'ADPSouple'
    dataset = ImageDataset(base_dir / 'example_images')
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn
    )

    logger = logging.getLogger('ADPSoupleExample')
    logger.info('-' * 70)
    logger.info(f'ADP Souple inference on {len(dataset)} example images')
    custom_timer = CustomTimer()
    custom_timer.start()

    with torch.inference_mode():
        for images, names, patches, sizes in dataloader:
            outputs = adpsouple(patches)
            process_and_save(images, names, outputs, sizes, output_dir)

    custom_timer.stop(logger, len(dataset))
    log_memory(logger)
    logger.info(f'Predictions saved in {output_dir}')


if __name__ == '__main__':
    main()

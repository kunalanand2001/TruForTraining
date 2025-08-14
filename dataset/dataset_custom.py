"""
This is a custom dataset loader that reads image and mask paths
from a provided text file list.

Each line in the text file should be in the format:
path/to/image.png path/to/mask.png
"""

from project_config import project_root
from dataset.AbstractDataset import AbstractDataset

import os
import numpy as np
from PIL import Image


class DatasetCustom(AbstractDataset):
    """
    Custom dataset loader that reads from a text file list.
    """

    def __init__(self, crop_size, grid_crop, img_list: str, max_dim=None, aug=None):
        super().__init__(crop_size, grid_crop, max_dim, aug=aug)
        
        # The text files contain paths relative to the project root,
        # so we set the root path to the project root itself.
        self._root_path = project_root
        
        # Read the list of image and mask paths from the provided text file.
        list_path = os.path.join(self._root_path, img_list)
        with open(list_path, "r") as f:
            # Assumes paths are separated by a space
            self.img_list = [t.strip().split(' ') for t in f.readlines()]
        
        print(f"Loaded {len(self.img_list)} image paths from {img_list}.")


    def get_img(self, index):
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        
        # The paths in the text file are already relative to the project root
        rgb_path, mask_path = self.img_list[index]
            
        if mask_path.lower() == 'none':
            mask = None
        else:
            full_mask_path = os.path.join(self._root_path, mask_path)
            mask = np.array(Image.open(full_mask_path).convert("L"))
            mask[mask > 0] = 1

        full_rgb_path = os.path.join(self._root_path, rgb_path)
        assert os.path.isfile(full_rgb_path), f"Image file not found: {full_rgb_path}"
        return self._create_tensor(mask=mask, rgb_path=full_rgb_path)

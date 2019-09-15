"""Human segmentation dataset classes"""
import os

import torch  # type: ignore
import torchvision  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image  # type: ignore

from lib import *  # type: ignore


class HumanSegmentationDatasetCSV(torch.utils.data.Dataset):
    """
    CSV human segmentation dataset
    
    Parameters
    ----------
    config : dict
        see configs/example.json for keys
        
    path : str
        path to dataset .csv file
        
    transform_flag : bool
        do or not augmentations [default False]
    """
    def __init__(self, config, path, transform_flag=False):
        self.annotation = pd.read_csv(path)
        self.prefix = config['prefix']

        self.names = self.annotation['name'].values
        self.ids = self.annotation['id'].values.astype(int)
        self.rle_masks = self.annotation['rle_mask'].values

        self.width = config['width']
        self.height = config['height']

        self.transform_flag = transform_flag
        self.transform = None

        self.color_jitter = not (
            'color_jitter' not in config or config['color_jitter'] == 0
        )

        if self.transform_flag:
            self.transform = list()

            self.transform.append(
                torchvision.transforms.Resize(
                    (self.height, self.width),
                    interpolation=3
                )
            )
            if self.color_jitter:
                self.transform.append(
                    torchvision.transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0
                    )
                )
            self.transform.append(
                torchvision.transforms.RandomHorizontalFlip()
            )
            self.transform.append(
                torchvision.transforms.ToTensor()
            )
            self.transform.append(
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            )

            self.transform = torchvision.transforms.Compose(self.transform)

        else:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(
                        (self.height, self.width),
                        interpolation=3
                    ),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ]
            )

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        name = self.names[idx]

        name = os.path.join(self.prefix, name)

        image = Image.open(name)
        image = self.transform(image)

        mask = decode_rle(self.rle_masks[idx])

        sample = (image, mask)

        return sample

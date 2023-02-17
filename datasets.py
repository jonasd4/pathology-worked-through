from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as plt
from typing import Optional
from torchvision.datasets import VisionDataset
import os
import pandas as pd
from PIL import Image
import torch

IMAGENET_NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
NORM = [[0.75104631, 0.47663395, 0.67639612], [0.04489756, 0.06757068, 0.07309701]]


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        result = img.convert("RGB")
    return result


class PatchDataset(VisionDataset):

    def __init__(self, split='df', *args, **kwargs):
        super().__init__(*args, **kwargs)
        split_path = os.path.join(self.root, split + '.csv')
        self.df = pd.read_csv(split_path)
        self.images_root = os.path.dirname(split_path)
        self.targets = self.df.label.values

    def __getitem__(self, index: int):
        entry = self.df.iloc[index]
        target = self.targets[index]
        file_path = entry['file_path']

        image_path = os.path.join(self.images_root, file_path)
        sample = pil_loader(image_path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.df)


class TissueDataModule(plt.LightningDataModule):

    def __init__(self, train_path, test_path, dl_kwargs, train_split='train', test_split='test',
                 image_size=224, augmentation=None):
        super().__init__()
        self.dl_kwargs = dl_kwargs

        if augmentation == 'v1':
            train_aug = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*IMAGENET_NORM)
            ])
        elif augmentation == 'v2':
            train_aug = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*IMAGENET_NORM)
            ])
        elif augmentation == 'v3':
            train_aug = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
                transforms.RandomEqualize(p=0.3),
                transforms.RandomApply([transforms.RandomGrayscale(3)], p=0.3),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*IMAGENET_NORM)
            ])
        else:
            raise ValueError()

        test_aug = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(*IMAGENET_NORM)
        ])

        self.train_ds = PatchDataset(root=train_path, split=train_split, transform=train_aug)
        self.test_ds = PatchDataset(root=test_path, split=test_split, transform=test_aug)

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        sampler = torch.utils.data.RandomSampler(data_source=self.train_ds,
                                                 replacement=False,
                                                 num_samples=5000)
        return DataLoader(self.train_ds, sampler=sampler, **self.dl_kwargs)
        # return DataLoader(self.train_ds, drop_last=False, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_ds, drop_last=False, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, drop_last=False, **self.dl_kwargs)

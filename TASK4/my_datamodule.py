# TUWIEN - WS2022 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL
import numpy as np


def pil_loader(p):
    return PIL.Image.open(p)


class DataModule:

    def __init__(self, data_dir: str = 'data/facemask', img_size: int = 64, batch_size: int = 32, augmented=False, num_workers=1, preload=False):
        """
        Initializes the DataModule
        data_dir: path to the directory of the data
        img_size: size of the images
        batch_size: number of images used for each iteration
        augmented: true if the data should be augmented
        gray: true if the data should be grayscaled
        num_workers: number of worker threads that load the data
        preload: true if the data should only be loaded once from disk
        """

        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augmented = augmented
        self.num_workers = num_workers

        self.train_dataset = FaceMaskDataset(self.data_dir + '/train', transform=self.get_transforms(
            train=True), label_transform=self.label_transform(), preload=preload)
        self.val_dataset = FaceMaskDataset(self.data_dir + '/val', transform=self.get_transforms(
        ), label_transform=self.label_transform(), preload=preload)
        self.test_dataset = FaceMaskDataset(self.data_dir + '/test', transform=self.get_transforms(
        ), label_transform=self.label_transform(), preload=preload)

    def label_transform(self):
        return torch.Tensor

    def get_transforms(self, train: bool = False):
        """
        returns transformations that should be applied to the dataset
        HINT: transforms.Compose([...]), transforms.ToTensor(),
        transforms.Resize((...)), transforms.RandomHorizontalFlip()
        and transforms.RandomAffine(...)
        train: boolean if true training transformations are returned
        if self.augmented and train is true add data augmentation
        """

        data_transforms = None

        # student code start

        # Resize to 64x64 pixel and normalize (transforms.ToTensor() normalizes to [0,1])
        data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])
        
        if self.augmented:
            data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=(0.5, 1), contrast=(0.9, 1), saturation=(0.5,0.7), hue=0), transforms.RandomRotation(degrees=33)])
        
        # student code end

        return data_transforms

    def train_dataloader(self):
        # returns the train dataloader
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # returns the value dataloader
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # returns the test dataloader
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=self.num_workers)


class FaceMaskDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, label_transform=None, preload=False):
        """
        Initializes the Face Mask Dataset
        data_dir: subdirectory of the facemask dataset
        transform: transformations for the dataset
        label_transform: transformations applied to the labels
        preload: true if the data should be loaded only once
        """

        self.preload = preload
        self.data_dir = data_dir
        self.face_paths = [
            f'{data_dir}/face/{name}' for name in os.listdir(f'{data_dir}/face')]
        self.mask_paths = [
            f'{data_dir}/mask/{name}' for name in os.listdir(f'{data_dir}/mask')]
        if preload:
            self.faces = [pil_loader(img_path)
                          for img_path in self.face_paths]
            self.masks = [pil_loader(img_path)
                          for img_path in self.mask_paths]
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        # Returns the length of the dataset
        return len(self.face_paths) + len(self.mask_paths)

    def __getitem__(self, idx: int):
        """
        Given an index, returns a sample of the dataset
        idx: index of the sample
        """
        if idx < len(self.face_paths):
            label = 0
            if self.preload:
                image = self.faces[idx]
            else:
                img_path = self.face_paths[idx]
        else:
            label = 1
            if self.preload:
                image = self.masks[idx - len(self.face_paths)]
            else:
                img_path = self.mask_paths[idx - len(self.face_paths)]

        if not self.preload:
            image = pil_loader(img_path)

        if self.label_transform:
            label = self.label_transform([label])
        if self.transform:
            image = self.transform(image)
        return image, label.float()

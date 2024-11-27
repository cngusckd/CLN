import numpy as np
import torch

from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

class numpyDataset(Dataset):
    def __init__(self, examples_path, labels_path, transform=None, target_transform=None):
        self.data = np.load(examples_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index, 0]  # Extract scalar from (1,) shape
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data)

class CifarDS:
    def __init__(self, directory: str = "./data/cifar10/"):

        # best means and stds for CIFAR10
        self.means = (0.4914, 0.4822, 0.4465)
        self.stds = (0.247, 0.243, 0.261)

        self.transform_basic = None
        self.transform_augment = None

        self.directory = directory

    def define_transforms(
        self, base_transforms: list = None, augmented_transforms: list = None
    ) -> None:
        if not base_transforms:  # use default
            base_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
            ]

        if not augmented_transforms:  # use default
            augmented_transforms = [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    hue=0.01, brightness=0.3, contrast=0.3, saturation=0.3
                ),
            ]
        augmented_transforms += base_transforms

        self.transform_basic = transforms.Compose(base_transforms)
        self.transform_augment = transforms.Compose(augmented_transforms)

    def get_train_gen(
        self,
        n_batches: int = None,
        batch_size: int = 64,
        base_transforms: list = None,
        augmented_transforms: list = None,
    ) -> DataLoader:
        self.define_transforms(base_transforms, augmented_transforms)
        # train_ds = CIFAR10(
        #     self.directory, train=True, download=True, transform=self.transform_augment
        # )
        train_ds = numpyDataset(
            examples_path = 'examples.npy',
            labels_path = 'labels.npy',
            # transform = self.transform_augment
        )
        if n_batches:
            train_ds.data = train_ds.data[: n_batches * batch_size]
        train_batch_gen = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        return train_batch_gen

    def get_valid_gen(
        self,
        n_batches: int = None,
        batch_size: int = 512,
        base_transforms: list = None,
        augmented_transforms: list = None,
    ) -> DataLoader:
        self.define_transforms(base_transforms, augmented_transforms)
        val_ds = CIFAR10(
            self.directory, train=False, download=True, transform=self.transform_basic
        )
        if n_batches:
            val_ds.data = val_ds.data[: n_batches * batch_size]
        val_batch_gen = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return val_batch_gen
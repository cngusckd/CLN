import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, Subset, random_split

class IncrementalMNIST(MNIST):
    def __init__(self, cfg, root='./data', train=True, transform=None, target_transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.cfg = cfg
        self.num_increments = cfg.num_increments
        self.batch_size = cfg.batch_size
        self.increment_type = cfg.cl_type

        # Split data indices based on the increment type
        if self.increment_type == 'cil':
            self.indices_by_class = [[] for _ in range(10)]
            self._split_indices_by_class()
            self.incremental_loaders = []
            self._create_class_incremental_loaders()
        elif self.increment_type == 'dil':
            self.incremental_loaders = []
            self._create_domain_incremental_loaders()
        else:
            raise ValueError("increment_type must be either 'class' or 'domain'.")

    def _split_indices_by_class(self):
        # Extract indices for each class
        for idx, (_, label) in enumerate(self):
            self.indices_by_class[label].append(idx)

    def _create_class_incremental_loaders(self):
        # Number of classes per increment
        classes_per_increment = 10 // self.num_increments

        # Create dataset for each increment
        for increment in range(self.num_increments):
            current_classes = range(increment * classes_per_increment, (increment + 1) * classes_per_increment)

            # Extract indices for the current increment
            current_indices = [idx for cls in current_classes for idx in self.indices_by_class[cls]]

            # Create a subset of the dataset
            subset = Subset(self, current_indices)

            # Create DataLoader
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False,
                                num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory)

            # Add to the list of incremental datasets
            self.incremental_loaders.append(loader)

    def _create_domain_incremental_loaders(self):
        # Split the dataset into domain increments (randomly split into num_increments parts)
        dataset_size = len(self)
        increment_size = dataset_size // self.num_increments
        lengths = [increment_size] * (self.num_increments - 1)
        lengths.append(dataset_size - sum(lengths))  # Include the remainder in the last increment

        subsets = random_split(self, lengths)

        # Create DataLoader for each subset
        for subset in subsets:
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False,
                                num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory)
            self.incremental_loaders.append(loader)

    def get_incremental_loader(self, increment_index):
        if increment_index < 0 or increment_index >= self.num_increments:
            raise ValueError(f"increment_index must be between 0 and {self.num_increments - 1}")
        
        return self.incremental_loaders[increment_index]

class IncrementalCIFAR10(CIFAR10):
    def __init__(self, cfg, root='./data', train=True, transform=None, target_transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.cfg = cfg
        self.num_increments = cfg.num_increments
        self.batch_size = cfg.batch_size
        self.increment_type = cfg.cl_type

        # Split data indices based on the increment type
        if self.increment_type == 'cil':
            self.indices_by_class = [[] for _ in range(10)]
            self._split_indices_by_class()
            self.incremental_loaders = []
            self._create_class_incremental_loaders()
        elif self.increment_type == 'dil':
            self.incremental_loaders = []
            self._create_domain_incremental_loaders()
        else:
            raise ValueError("increment_type must be either 'cil' or 'dil'.")

    def _split_indices_by_class(self):
        # Extract indices for each class
        for idx, (_, label) in enumerate(self):
            self.indices_by_class[label].append(idx)

    def _create_class_incremental_loaders(self):
        # Number of classes per increment
        classes_per_increment = 10 // self.num_increments

        # Create dataset for each increment
        for increment in range(self.num_increments):
            current_classes = range(increment * classes_per_increment, (increment + 1) * classes_per_increment)

            # Extract indices for the current increment
            current_indices = [idx for cls in current_classes for idx in self.indices_by_class[cls]]

            # Create a subset of the dataset
            subset = Subset(self, current_indices)

            # Create DataLoader
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False)

            # Add to the list of incremental datasets
            self.incremental_loaders.append(loader)

    def _create_domain_incremental_loaders(self):
        # Split the dataset into domain increments (randomly split into num_increments parts)
        dataset_size = len(self)
        increment_size = dataset_size // self.num_increments
        lengths = [increment_size] * (self.num_increments - 1)
        lengths.append(dataset_size - sum(lengths))  # Include the remainder in the last increment

        subsets = random_split(self, lengths)

        # Create DataLoader for each subset
        for subset in subsets:
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False)
            self.incremental_loaders.append(loader)

    def get_incremental_loader(self, increment_index):
        if increment_index < 0 or increment_index >= self.num_increments:
            raise ValueError(f"increment_index must be between 0 and {self.num_increments - 1}")
        
        return self.incremental_loaders[increment_index]

class IncrementalCIFAR100(CIFAR100):
    def __init__(self, cfg, root='./data', train=True, transform=None, target_transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.cfg = cfg
        self.num_increments = cfg.num_increments
        self.batch_size = cfg.batch_size
        self.increment_type = cfg.cl_type

        # Split data indices based on the increment type
        if self.increment_type == 'cil':
            self.indices_by_class = [[] for _ in range(100)]
            self._split_indices_by_class()
            self.incremental_loaders = []
            self._create_class_incremental_loaders()
        elif self.increment_type == 'dil':
            self.incremental_loaders = []
            self._create_domain_incremental_loaders()
        else:
            raise ValueError("increment_type must be either 'cil' or 'dil'.")

    def _split_indices_by_class(self):
        # Extract indices for each class
        for idx, (_, label) in enumerate(self):
            self.indices_by_class[label].append(idx)

    def _create_class_incremental_loaders(self):
        # Number of classes per increment
        classes_per_increment = 100 // self.num_increments

        # Create dataset for each increment
        for increment in range(self.num_increments):
            current_classes = range(increment * classes_per_increment, (increment + 1) * classes_per_increment)

            # Extract indices for the current increment
            current_indices = [idx for cls in current_classes for idx in self.indices_by_class[cls]]

            # Create a subset of the dataset
            subset = Subset(self, current_indices)

            # Create DataLoader
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False)

            # Add to the list of incremental datasets
            self.incremental_loaders.append(loader)

    def _create_domain_incremental_loaders(self):
        # Split the dataset into domain increments (randomly split into num_increments parts)
        dataset_size = len(self)
        increment_size = dataset_size // self.num_increments
        lengths = [increment_size] * (self.num_increments - 1)
        lengths.append(dataset_size - sum(lengths))  # Include the remainder in the last increment

        subsets = random_split(self, lengths)

        # Create DataLoader for each subset
        for subset in subsets:
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False)
            self.incremental_loaders.append(loader)

    def get_incremental_loader(self, increment_index):
        if increment_index < 0 or increment_index >= self.num_increments:
            raise ValueError(f"increment_index must be between 0 and {self.num_increments - 1}")
        
        return self.incremental_loaders[increment_index]

class CustomDataloader:
    def __init__(self, root='./data/tiny_imagenet/tiny-imagenet-200', batch_size=32, num_workers=4, pin_memory=True):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Define the transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images to 64x64
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Create datasets
        self.train_dataset = ImageFolder(root=os.path.join(self.root, 'train'), transform=self.transform)
        self.val_dataset = ImageFolder(root=os.path.join(self.root, 'val'), transform=self.transform)

class IncrementalCustomDataloader(CustomDataloader):
    def __init__(self, cfg, root='./data/tiny_imagenet/tiny-imagenet-200', train=True):
        super().__init__(root=root, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        
        self.cfg = cfg
        self.num_increments = cfg.num_increments
        self.increment_type = cfg.cl_type
        self.train = train

        # Choose dataset based on train flag
        self.dataset = self.train_dataset if train else self.val_dataset

        # Split data indices based on the increment type
        if self.increment_type == 'cil':
            self.indices_by_class = [[] for _ in range(len(self.dataset.classes))]
            self._split_indices_by_class()
            self.incremental_loaders = []
            self._create_class_incremental_loaders()
        elif self.increment_type == 'dil':
            self.incremental_loaders = []
            self._create_domain_incremental_loaders()
        else:
            raise ValueError("increment_type must be either 'cil' or 'dil'.")

    def _split_indices_by_class(self):
        # Extract indices for each class
        for idx, (_, label) in enumerate(self.dataset):
            self.indices_by_class[label].append(idx)

    def _create_class_incremental_loaders(self):
        # Number of classes per increment
        classes_per_increment = len(self.dataset.classes) // self.num_increments

        # Create dataset for each increment
        for increment in range(self.num_increments):
            current_classes = range(increment * classes_per_increment, (increment + 1) * classes_per_increment)

            # Extract indices for the current increment
            current_indices = [idx for cls in current_classes for idx in self.indices_by_class[cls]]

            # Create a subset of the dataset
            subset = Subset(self.dataset, current_indices)

            # Create DataLoader
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False,
                                num_workers=self.num_workers, pin_memory=self.pin_memory)

            # Add to the list of incremental datasets
            self.incremental_loaders.append(loader)

    def _create_domain_incremental_loaders(self):
        # Split the dataset into domain increments (randomly split into num_increments parts)
        dataset_size = len(self.dataset)
        increment_size = dataset_size // self.num_increments
        lengths = [increment_size] * (self.num_increments - 1)
        lengths.append(dataset_size - sum(lengths))  # Include the remainder in the last increment

        subsets = random_split(self.dataset, lengths)

        # Create DataLoader for each subset
        for subset in subsets:
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False,
                                num_workers=self.num_workers, pin_memory=self.pin_memory)
            self.incremental_loaders.append(loader)

    def get_incremental_loader(self, increment_index):
        if increment_index < 0 or increment_index >= self.num_increments:
            raise ValueError(f"increment_index must be between 0 and {self.num_increments - 1}")
        
        return self.incremental_loaders[increment_index]
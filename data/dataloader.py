import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
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

# Example usage
if __name__ == "__main__":
    # Create a Class-Incremental Learning dataset for training
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Class-Incremental Learning example
    print("Class-Incremental Learning Example:")
    class_incremental_mnist = IncrementalMNIST(root='./data', 
                                               train=True, 
                                               transform=transform, 
                                               num_increments=5, 
                                               batch_size=64, 
                                               increment_type='class')

    for i in range(5):
        train_loader = class_incremental_mnist.get_incremental_loader(i)
        for data, label in train_loader:
            print(label)
            break

    # Domain-Incremental Learning dataset creation
    print("\nDomain-Incremental Learning Example:")
    domain_incremental_mnist = IncrementalMNIST(root='./data', 
                                                train=True, 
                                                transform=transform, 
                                                num_increments=5, 
                                                batch_size=64, 
                                                increment_type='domain')

    for i in range(5):
        train_loader = domain_incremental_mnist.get_incremental_loader(i)
        for data, label in train_loader:
            print(label)
            break
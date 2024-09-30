import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, random_split

class IncrementalMNIST(MNIST):
    def __init__(self, root='./data', train=True, transform=None, target_transform=None, download=True,
                 num_increments=5, batch_size=64, increment_type='class'):
        # MNIST 초기화
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        self.num_increments = num_increments
        self.batch_size = batch_size
        self.increment_type = increment_type

        # 데이터 인덱스 분할
        if self.increment_type == 'class':
            # 클래스별 인덱스 나누기
            self.indices_by_class = [[] for _ in range(10)]
            self._split_indices_by_class()
            self.incremental_loaders = []
            self._create_class_incremental_loaders()
        elif self.increment_type == 'domain':
            # 도메인별 분할 (랜덤하게 num_increments개로 나눔)
            self.incremental_loaders = []
            self._create_domain_incremental_loaders()
        else:
            raise ValueError("increment_type must be either 'class' or 'domain'.")

    def _split_indices_by_class(self):
        # 각 클래스별 인덱스 추출
        for idx, (_, label) in enumerate(self):
            self.indices_by_class[label].append(idx)

    def _create_class_incremental_loaders(self):
        # 클래스당 증분의 개수
        classes_per_increment = 10 // self.num_increments

        # 각 증분 단계별로 데이터셋을 구성
        for increment in range(self.num_increments):
            current_classes = range(increment * classes_per_increment, (increment + 1) * classes_per_increment)

            # 현재 증분에서 사용할 인덱스 추출
            current_indices = [idx for cls in current_classes for idx in self.indices_by_class[cls]]

            # Subset으로 데이터셋 구성
            subset = Subset(self, current_indices)

            # DataLoader 생성
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False)

            # 증분 데이터셋 리스트에 추가
            self.incremental_loaders.append(loader)

    def _create_domain_incremental_loaders(self):
        # 데이터셋을 도메인 증분으로 분할 (랜덤하게 num_increments개로 분할)
        dataset_size = len(self)
        increment_size = dataset_size // self.num_increments
        lengths = [increment_size] * (self.num_increments - 1)
        lengths.append(dataset_size - sum(lengths))  # 마지막 증분은 나머지를 포함

        subsets = random_split(self, lengths)

        # 각 Subset에 대해 DataLoader 생성
        for subset in subsets:
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True if self.train else False)
            self.incremental_loaders.append(loader)

    def get_incremental_loader(self, increment_index):
        if increment_index < 0 or increment_index >= self.num_increments:
            raise ValueError(f"increment_index must be between 0 and {self.num_increments - 1}")
        
        return self.incremental_loaders[increment_index]

# 사용 예시
if __name__ == "__main__":
    # 학습 데이터용 Class-Incremental Learning 데이터셋 생성
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Class-Incremental Learning 예시
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

    # Domain-Incremental Learning 데이터셋 생성
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

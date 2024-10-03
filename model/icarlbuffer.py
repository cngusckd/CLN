import torch
import numpy as np
import random

class ICaRLBUFFER:
    '''
    iCaRL 메모리 버퍼
    '''
    
    def __init__(self, buffer_memory_size, image_shape) -> None:
        # 버퍼 사이즈에 대해 버퍼 초기화 진행
        self.buffer_memory_size = buffer_memory_size
        self.image_shape = image_shape
        self.seen_classes = set()
        
        self.memory_size_per_class = 0
        self.examples = torch.zeros((self.buffer_memory_size, 3, self.image_shape[0], self.image_shape[1]), dtype = torch.float32)
        self.labels = torch.zeros((self.buffer_memory_size, 1), dtype = torch.int64)
        
    def __len__(self):
        return min(self.num_seen_examples, self.buffer_memory_size)
    
    def update_seen_classes(self, train_loader):
        # train_loader의 레이블값을 참조해 현재 어떤 label을 봤는 지 체크
        
        for _, labels in train_loader:
            labels = labels.tolist()
            self.seen_classes.update(labels)
        
        self.memory_size_per_class = self.buffer_memory_size // len(self.seen_classes)
        
        
        

if __name__ == '__main__':
    # ICaRLBUFFER TEST
    import sys
    sys.path.append('/root/CL/data')
    from torchvision import transforms
    from dataloader import IncrementalMNIST
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1채널 이미지를 3채널로 복사
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train = IncrementalMNIST(root = './data',
                             transform = transform
                             )
    train_loader = train.get_incremental_loader(0)
    
    buffer = ICaRLBUFFER(buffer_memory_size= 500, image_shape=(64, 64))
    
    buffer.update_seen_classes(train_loader)
    
    sys.path.remove('/root/CL/data')
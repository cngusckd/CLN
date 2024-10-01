import torch

from torchvision import transforms
from tqdm import tqdm

from model.er import ER
from data.dataloader import IncrementalMNIST

if __name__ == '__main__':
    
    _NUM_INCREMENTS = 5
    _NCLASSES = 10
    _EPOCH = 10
    _BACH_SIZE = 64
    _BUFFER_BATCH_SIZE = 64
    _DEVICE = torch.device('cuda')
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1채널 이미지를 3채널로 복사
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    cil_mnist_train = IncrementalMNIST(root = './data',
                                       train = True,
                                       transform = transform,
                                       num_increments = _NUM_INCREMENTS,
                                       batch_size = _BACH_SIZE,
                                       increment_type = 'class')
    cil_mnist_test = IncrementalMNIST(root = './data',
                                       train = False,
                                       transform = transform,
                                       num_increments = _NUM_INCREMENTS,
                                       batch_size = _BACH_SIZE,
                                       increment_type = 'class')
    
    cl_model = ER(nclasses = _NCLASSES,
                  buffer_memory_size = 1000,
                  buffer_batch_size = _BUFFER_BATCH_SIZE,
                  input_size = (28,28),
                  _DEVICE = _DEVICE)
    
    val_loader_list = []
    for _idx in range(_NUM_INCREMENTS):
        val_loader_list.append(cil_mnist_test.get_incremental_loader(_idx))
    
    for _incremental_time in range(_NUM_INCREMENTS):
        
        train_loader = cil_mnist_train.get_incremental_loader(_incremental_time)

        if _incremental_time == 0:
            for epoch in range(_EPOCH):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total = len(train_loader),
                                           ncols = 150):

                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels)
        else:
            for epoch in range(_EPOCH):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total = len(train_loader),
                                           ncols = 150):
                    
                    sampled_inputs, sampled_labels = cl_model.buffer_sampling()
                    inputs = torch.cat((inputs, sampled_inputs))
                    labels = torch.cat((labels, sampled_labels))
                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels)
        
        for _incremental_time, test_loader in enumerate(val_loader_list[:_incremental_time+1]):
            test_reulsts = cl_model.eval(test_loader, _incremental_time)
            print(test_reulsts)
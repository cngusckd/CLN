import torch

from torchvision import transforms
from tqdm import tqdm
from argparse import ArgumentParser


from model.er import ER
from model.icarl import iCaRL
from data.dataloader import IncrementalMNIST

def pasre_arg():
    
    cfg = ArgumentParser(description = "CHU CL Framework",
                            allow_abbrev = False)
    
    # CL Experiments Settings
    cfg.add_argument('--dataset', type = str, default = 'mnist',
                     help = 'experiment dataset', choices = ['mnist'])
    cfg.add_argument('--cl_type', type = str, default = 'cil',
                     help = 'CL exepriment type', choices = ['cil, dil'])
    cfg.add_argument('--nclasses', type = int, default = 10,
                     help = 'nclasses of dataset')
    cfg.add_argument('--num_increments', type = int, default = 5,
                     help = 'task num of CL')
    cfg.add_argument('--device', type = str, default='cuda',
                     help = 'deep learing devices', choices = ['cpu', 'cuda'])
    # check CL tasks
    temp = cfg.parse_args()
    if temp.nclasses % temp.num_increments != 0:
        raise "cant divide into CL tasks!"
    cfg.add_argument('--epoch', type = int, default = 10,
                     help = 'epochs per task')
    cfg.add_argument('--batch_size', type = int, default = 64,
                     help = 'batch size for current data stream, when incremental learning is adoptted, \
                         total batch size is batch_size + buffer_batch_size')
    cfg.add_argument('--buffer_batch_size', type = int, default = 64,
                     help = 'batch size for sampled buffer data, when incremental learning is adoptted, \
                         total batch size is batch_size + buffer_batch_size')
    
    return cfg.parse_args()


def er_train_example(cfg, train, test):
    
    cl_model = ER(nclasses = cfg.nclasses,
                  buffer_memory_size = 1000,
                  buffer_batch_size = cfg.buffer_batch_size,
                  image_shape = (28,28),
                  _DEVICE = torch.device(cfg.device))
    
    val_loader_list = []
    for _idx in range(cfg.num_increments):
        val_loader_list.append(test.get_incremental_loader(_idx))
    
    for _incremental_time in range(cfg.num_increments):
        
        train_loader = train.get_incremental_loader(_incremental_time)

        if _incremental_time == 0:
            for epoch in range(cfg.epoch):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total = len(train_loader),
                                           ncols = 150):

                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels)
        else:
            for epoch in range(cfg.epoch):
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

def icarl_train_example(cfg, train, test):
    
    
    cl_model = iCaRL(nclasses = cfg.nclasses,
                  buffer_memory_size = 1000,
                  buffer_batch_size = cfg.buffer_batch_size,
                  image_shape = (28,28),
                  _DEVICE = torch.device(cfg.device))
    
    val_loader_list = []
    for _idx in range(cfg.num_increments):
        val_loader_list.append(test.get_incremental_loader(_idx))
    
    for _incremental_time in range(cfg.num_increments):
        
        train_loader = train.get_incremental_loader(_incremental_time)

        if _incremental_time == 0:
            
            cl_model.buffer.update_seen_classes(train_loader)
            exit()
            for epoch in range(cfg.epoch):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total = len(train_loader),
                                           ncols = 150):

                    cl_model.observe(inputs, labels)
        else:
            for epoch in range(cfg.epoch):
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


if __name__ == '__main__':
    
    cfg = pasre_arg()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1채널 이미지를 3채널로 복사
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    cil_mnist_train = IncrementalMNIST(root = './data',
                                       train = True,
                                       transform = transform,
                                       num_increments = cfg.num_increments,
                                       batch_size = cfg.batch_size,
                                       increment_type = cfg.cl_type)
    cil_mnist_test = IncrementalMNIST(root = './data',
                                       train = False,
                                       transform = transform,
                                       num_increments = cfg.num_increments,
                                       batch_size = cfg.batch_size,
                                       increment_type = cfg.cl_type)
    
    icarl_train_example(cfg, cil_mnist_train, cil_mnist_test)
import torch

from torchvision import transforms
from tqdm import tqdm
from argparse import ArgumentParser


from model.er import ER, er_train_example
from model.der import DER, der_train_example
from data.dataloader import IncrementalMNIST

def pasre_arg():
    
    cfg = ArgumentParser(description = "CHU CL Framework",
                            allow_abbrev = False)
    
    # CL Experiments Settings
    cfg.add_argument('--dataset', type = str, default = 'mnist',
                     help = 'experiment dataset', choices = ['mnist'])
    cfg.add_argument('--cl_type', type = str, default = 'cil',
                     help = 'CL exepriment type', choices = ['cil', 'dil'])
    cfg.add_argument('--nclasses', type = int, default = 10,
                     help = 'nclasses of dataset')
    cfg.add_argument('--num_increments', type = int, default = 5,
                     help = 'task num of CL')
    cfg.add_argument('--device', type = str, default='cuda',
                     help = 'deep learing devices', choices = ['cpu', 'cuda'])
    cfg.add_argument('--epoch', type = int, default = 10,
                     help = 'epochs per task')
    cfg.add_argument('--batch_size', type = int, default = 64,
                     help = 'batch size for current data stream, when incremental learning is adoptted, \
                         total batch size is batch_size + buffer_batch_size')
    cfg.add_argument('--buffer_batch_size', type = int, default = 64,
                     help = 'batch size for sampled buffer data, when incremental learning is adoptted, \
                         total batch size is batch_size + buffer_batch_size')
    cfg.add_argument('--buffer_memory_size', type = int, default = 200,
                     help = 'batch size for sampled buffer data, when incremental learning is adoptted, \
                         total batch size is batch_size + buffer_batch_size')
    # check CL tasks
    temp = cfg.parse_args()
    if temp.nclasses % temp.num_increments != 0:
        raise "cant divide into CL tasks!"
    
    return cfg.parse_args()


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
    er_avg_acc = er_train_example(cfg, cil_mnist_train, cil_mnist_test)
    der_avg_acc = der_train_example(cfg, cil_mnist_train, cil_mnist_test)
    
    experimental_output = dict()
    experimental_output['ER'] = er_avg_acc
    experimental_output['DER'] = der_avg_acc
    
    print(f'\n\n\n {experimental_output} \n\n\n')
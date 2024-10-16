import json
import os
import pytz

from torchvision import transforms
from argparse import ArgumentParser
from datetime import datetime

from model.er import er_train_example
from model.der import der_train_example, derpp_train_example
from model.er_ace import er_ace_train_example
from model.gss import er_gss_train_example
from model.mir import er_mir_train_example

from data.dataloader import IncrementalMNIST, IncrementalCIFAR10, IncrementalCIFAR100

def pasre_arg():
    
    cfg = ArgumentParser(description = "CHU CL Framework",
                            allow_abbrev = False)
    
    # CL Experiments Settings
    cfg.add_argument('--dataset', type = str, default = 'mnist',
                     help = 'experiment dataset', choices = ['mnist', 'cifar10', 'cifar100'])
    cfg.add_argument('--image_shape', type = set, default = (32,32),
                     help = 'image_shpae of dataset')
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
    cfg.add_argument('--buffer_memory_size', type = int, default = 500,
                     help = 'batch size for sampled buffer data, when incremental learning is adoptted, \
                         total batch size is batch_size + buffer_batch_size')
    # check CL tasks
    temp = cfg.parse_args()
    if temp.nclasses % temp.num_increments != 0:
        raise "cant divide into CL tasks!"
    
    return cfg.parse_args()


if __name__ == '__main__':
    
    cfg = pasre_arg()
    
    if cfg.dataset == 'cifar10':
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        cil_train = IncrementalCIFAR10(root = './data',
                                           train = True,
                                           transform = transform,
                                           num_increments = cfg.num_increments,
                                           batch_size = cfg.batch_size,
                                           increment_type = cfg.cl_type)
        cil_test = IncrementalCIFAR10(root = './data',
                                           train = False,
                                           transform = transform,
                                           num_increments = cfg.num_increments,
                                           batch_size = cfg.batch_size,
                                           increment_type = cfg.cl_type)
    elif cfg.dataset == 'cifar100':
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        cil_train = IncrementalCIFAR100(root = './data',
                                           train = True,
                                           transform = transform,
                                           num_increments = cfg.num_increments,
                                           batch_size = cfg.batch_size,
                                           increment_type = cfg.cl_type)
        cil_test = IncrementalCIFAR100(root = './data',
                                           train = False,
                                           transform = transform,
                                           num_increments = cfg.num_increments,
                                           batch_size = cfg.batch_size,
                                           increment_type = cfg.cl_type)
    elif cfg.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1채널 이미지를 3채널로 복사, MNIST 학습에 사용
            transforms.Normalize((0.5,), (0.5,))
        ])
        cfg.image_shape = [28, 28]
        cil_train = IncrementalMNIST(root = './data',
                                           train = True,
                                           transform = transform,
                                           num_increments = cfg.num_increments,
                                           batch_size = cfg.batch_size,
                                           increment_type = cfg.cl_type)
        cil_test = IncrementalMNIST(root = './data',
                                           train = False,
                                           transform = transform,
                                           num_increments = cfg.num_increments,
                                           batch_size = cfg.batch_size,
                                           increment_type = cfg.cl_type)
        
    experimental_output = dict()
    experimental_output['CONFIGS'] = vars(cfg)
    local_tz = pytz.timezone('Asia/Seoul')
    saved_dir = f'experiment_outputs_test/{datetime.now(local_tz).strftime("%Y%m%d%H%M")}'
    os.makedirs(saved_dir, exist_ok = True)
    with open(os.path.join(saved_dir, f'{cfg.dataset}_{cfg.cl_type}.json'), mode = 'w') as f:
        json.dump(experimental_output, f, indent = 4)
    
    der_avg_acc = der_train_example(cfg, cil_train, cil_test)
    derpp_avg_acc = derpp_train_example(cfg, cil_train, cil_test)
    er_avg_acc = er_train_example(cfg, cil_train, cil_test) # check Done
    er_mir_avg_acc = er_mir_train_example(cfg, cil_test, cil_test)
    er_ace_avg_acc = er_ace_train_example(cfg, cil_train, cil_test)
    er_gss_avg_acc = er_gss_train_example(cfg, cil_train, cil_test)
    
    experimental_output['ER-MIR'] = er_mir_avg_acc
    experimental_output['ER-GSS'] = er_gss_avg_acc
    experimental_output['ER'] = er_avg_acc
    experimental_output['DER'] = der_avg_acc
    experimental_output['DER++'] = derpp_avg_acc
    experimental_output['ER-ACE'] = er_ace_avg_acc
    
    print(f'\n\n\n {experimental_output} \n\n\n')
    
    with open(os.path.join(saved_dir, f'{cfg.dataset}_{cfg.cl_type}.json'), mode = 'w') as f:
        json.dump(experimental_output, f, indent = 4)
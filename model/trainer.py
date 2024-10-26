from torchvision import transforms

from data.dataloader import *

def build_model(cfg):
    
    if cfg.model == 'er':
        from model.er import ER
        
        return ER(cfg)
    elif cfg.model == 'er_ace':
        from model.er_ace import ER_ACE
        
        return ER_ACE(cfg)
    elif cfg.model in ['der', 'der++']:
        from model.der import DER
        
        return DER(cfg)
    else:
        raise NotImplementedError

def build_data(cfg):
    
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
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        '''
            1채널 이미지를 3채널로 복사, MNIST 학습에 사용, 
            기반 ResNet의 input channel이 3으로 설정되어 있기 때문, 
            1채널을 사용하기 위해서는 기반 ResNet 수정 필요
        '''
        
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
    else:
        raise NotImplementedError
    
    return cil_train, cil_test
    

class CL_Trainer:
    
    def __init__(self, cfg):
        
        self.train_loader, self.test_loader = build_data(cfg)
        # build by experimental settings in main.py
        
        self.cl_model = build_model(cfg)
        # provide ER based CL model
        # ER / DER / DER++ / ER-ACE
        
        self.cfg = cfg
        
    def begin_continual_learning(self):
        
        exp_outputs = []
        '''
        need to implement exp_outputs
        e.g.) Accuracy, AUROC, Confusion Matrix, Buffer Distribution, etc.....
        '''
        val_loader_list = []
        
        for task_idx in range(self.cfg.num_increments):
            
            val_loader_list.append(self.test_loader.get_incremental_loader(task_idx))
            # load test_loader until task t (current task)
            
            train_loader = self.train_loader.get_incremental_loader(task_idx)
            
            self.cl_model.train_task(train_loader)
            # continual learning with current task
            
            for task_index, val_loader in enumerate(val_loader_list):
                print(self.cl_model.eval_task(val_loader, task_index))
                # evaluate model with val_loader until current task
            
            
        
        return exp_outputs
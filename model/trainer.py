import wandb

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
        cil_train = IncrementalCIFAR10(cfg, root='./data', train=True, transform=transform)
        cil_test = IncrementalCIFAR10(cfg, root='./data', train=False, transform=transform)
    elif cfg.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        cil_train = IncrementalCIFAR100(cfg, root='./data', train=True, transform=transform)
        cil_test = IncrementalCIFAR100(cfg, root='./data', train=False, transform=transform)
    elif cfg.dataset == 'mnist':
        cfg.image_shape = (28,28) # for mnist, the image shape is 28x28
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel image to 3-channel
            transforms.Normalize((0.5,), (0.5,))
        ])
        '''
        The MNIST dataset is originally in 1-channel (grayscale).
        To use it with models like ResNet, which expect 3-channel input,
        we replicate the single channel three times.
        This allows the model to process MNIST images without modifying the network architecture.
        '''
        cil_train = IncrementalMNIST(cfg, root='./data', train=True, transform=transform)
        cil_test = IncrementalMNIST(cfg, root='./data', train=False, transform=transform)
    elif cfg.dataset == 'custom_dataset':
        transform = transforms.Compose([
            transforms.Resize((cfg.image_shape[0], cfg.image_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        cil_train = IncrementalCustomDataloader(cfg, root='./data/tiny_imagenet/tiny-imagenet-200', transform = transform, train=True)
        cil_test = IncrementalCustomDataloader(cfg, root='./data/tiny_imagenet/tiny-imagenet-200', transform = transform, train=False)
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
        
    def begin_continual_learning(self):
        
        exp_outputs = []
        '''
        need to implement exp_outputs
        e.g.) Accuracy, AUROC, Confusion Matrix, Buffer Distribution, etc.....
        '''        
        
        val_loader_list = []
        
        for task_idx in range(self.cl_model.cfg.num_increments):
            
            val_loader_list.append(self.test_loader.get_incremental_loader(task_idx))
            # load test_loader until task t (current task)
            
            train_loader = self.train_loader.get_incremental_loader(task_idx)
            
            self.cl_model.train_task(train_loader)
            # continual learning with current task
            
            # evaluate model with val_loader until current task
            val_acc, val_loss, auroc, conf_matrix, all_labels, all_probs = self.cl_model.eval_task(val_loader_list)
            if self.cl_model.cfg.wandb:
                self.cl_model.wandb_eval_logger(val_acc, val_loss, auroc, conf_matrix, all_labels, all_probs, task_idx)

        return exp_outputs
import torch
import torch.nn as nn

from model.resnet import resnet18
from model.buffer import DefaultBuffer

class CL_MODEL(nn.Module):
    
    def __init__(self,
                 cfg
                 ):
        super().__init__()
        
        self.cfg = cfg
        
        self.device = cfg.device
        # device for training
        
        self.backbone = resnet18(nclasses = cfg.nclasses, nf = cfg.image_shape[0]).to(self.device)
        # backbone networks, now only support resnet18
        # you can modify to resnet34,50,101,121 from model/resnet.py
        # nf : input size(transformed input size)
        
        self.optimizer = self.get_optimizer()
        self.loss = self.get_loss_func()
        # optmizer & criteria
        
        self.current_task_index = 0
        # check current task_index for continual learning
        
        self.buffer = DefaultBuffer(cfg)
        # Buffer for continual learning
        
        
    def get_parameters(self):
        
        return self.backbone.parameters()
    
    def get_optimizer(self): # default settings
        
        if self.cfg.optim == 'sgd':
            return torch.optim.SGD(params = self.get_parameters(),
                                   lr = self.cfg.lr,
                                   momentum = self.cfg.momentum)
        else:
            raise NotImplementedError
        
    def get_loss_func(self):
        
        return torch.nn.CrossEntropyLoss()
    
    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.

        Returns:
            gradients tensor
        """
        grads = []
        for pp in list(self.backbone.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    
    def train_task():
        
        raise NotImplementedError
    
    def eval_task():
        
        raise NotImplementedError
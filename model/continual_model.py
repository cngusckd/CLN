import torch
import torch.nn as nn

from model.resnet import resnet18
from model.buffer import BUFFER

class CL_MODEL(nn.Module):
    
    def __init__(self,
                 nclasses,
                 buffer_memory_size,
                 buffer_batch_size,
                 image_shape,
                 _DEVICE
                 ):
        super().__init__()
        
        # 학습에 사용할 device 초기화
        self._DEVICE = _DEVICE
        
        # 버퍼 크기로 초기화
        # buffer 관련 연산들은 cpu에서 진행
        self.buffer = BUFFER(buffer_memory_size = buffer_memory_size, image_shape = image_shape)
        
        
        self.backbone = resnet18(nclasses = nclasses, nf = image_shape[0]).to(self._DEVICE)
        # 그냥 임시로 class수를 정해놓음, 따로 정의 필요
        # nf : input size의 크기(transformed input size)
        
        self.buffer_batch_size = buffer_batch_size
        # minibatch + _BUFFER_BATCH = TOTAL minibatch
        
        self.optimizer = self.get_optimizer()
        self.loss = self.get_loss_func()
        
    def get_parameters(self):
        return self.backbone.parameters()
    
    def get_optimizer(self):
        return torch.optim.SGD(params = self.get_parameters(),
                                lr = 1e-3,
                                momentum = 9e-1)
    
    def get_loss_func(self):
        return torch.nn.CrossEntropyLoss()
    
    def train_with_data_loader():
        pass
    
    def eval():
        pass
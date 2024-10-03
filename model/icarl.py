import torch
import torch.nn as nn
import torch.nn.functional as F

from model.continual_model import CL_MODEL
from model.icarlbuffer import ICaRLBUFFER

# CL_MODEL 클래스를 상속하여 iCaRL 모델 구현
class iCaRL(CL_MODEL):
    
    def __init__(self,
                 nclasses,
                 buffer_memory_size,
                 buffer_batch_size,
                 image_shape,
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)


        # 지금까지 학습한 class
        self.nclasses = nclasses
        self.set_of_seen_labels = set()
        self.buffer = ICaRLBUFFER(buffer_memory_size = buffer_memory_size,
                                  image_shape = image_shape)
        self._DEVICE = _DEVICE
        self.class_means = {}
    
    def loss(pred, y): # binary_cross_entropy
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()
     
    def observe(self, inputs, labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(inputs, outputs)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def compute_class_mean(self):
        
        seen_classes = self.buffer.seen_classes # 현재까지 본 레이블들
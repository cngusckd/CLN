import torch

from continual_model import CL_MODEL

class ER(CL_MODEL):
    
    def __init__(self, 
                 nclasses, 
                 buffer_memory_size, 
                 buffer_batch_size, 
                 input_size, 
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, input_size, _DEVICE)
        
        self.device = _DEVICE
        
if __name__=='__main__':
    
    continual_model = CL_MODEL(10, 100, 10, (64, 64), _DEVICE = torch.device('cuda'))
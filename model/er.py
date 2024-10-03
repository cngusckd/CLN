import torch

from torchvision import transforms
from tqdm import tqdm

from model.continual_model import CL_MODEL

class ER(CL_MODEL):
    
    def __init__(self, 
                 nclasses, 
                 buffer_memory_size, 
                 buffer_batch_size, 
                 image_shape, 
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)
        
        self._DEVICE = _DEVICE
    
    def observe(self, inputs, labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def buffer_update(self, inputs, labels):
        
        for input_data, input_label in zip(inputs, labels):
            self.buffer.update(input_data = input_data,
                               input_label = input_label)
    
    def buffer_sampling(self):
        
        if self.buffer.num_seen_examples > 0:
            sampled_inputs, sampled_labels = self.buffer.get_data(self.buffer_batch_size)
            
        return sampled_inputs, sampled_labels
    
    def eval(self, val_loader, _incremental_time):
    
        temp = {}
        
        self.backbone.eval()
        val_loss = 0
        val_acc = 0
        
        for inputs, labels in val_loader:
            
            inputs, labels, = inputs.to(self._DEVICE), labels.to(self._DEVICE)
            
            outputs = self.backbone(inputs)
            loss = self.loss(outputs, labels)
            
            val_loss += (loss.item() / len(inputs))
            
            pred = outputs.argmax(dim=1)
            
            val_acc += (pred == labels).float().sum()
        
        val_acc /= len(val_loader.dataset)
        torch.cuda.empty_cache()
        temp[f'Task_{_incremental_time}_EVAL_ACC'] = val_acc.item()
        temp[f'Task_{_incremental_time}_EVAL_LOSS'] = val_loss
        self.backbone.train()
        
        return temp
        
        
'''
# Example Code
if __name__ == '__main__':
    
    cfg = pasre_arg()
    
    cfg.num_increments = 5
    _EPOCH = 10
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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
    
    cl_model = ER(nclasses = cfg.nclasses,
                  buffer_memory_size = 1000,
                  buffer_batch_size = cfg.buffer_batch_size,
                  input_size = (28,28),
                  _DEVICE = torch.device(cfg.device))
    
    val_loader_list = []
    for _idx in range(cfg.num_increments):
        val_loader_list.append(cil_mnist_test.get_incremental_loader(_idx))
    
    for _incremental_time in range(cfg.num_increments):
        
        train_loader = cil_mnist_train.get_incremental_loader(_incremental_time)

        if _incremental_time == 0:
            for epoch in range(cfg.epoch):
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
'''
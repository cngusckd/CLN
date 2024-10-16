import torch
import torch.nn as nn
import copy

from torchvision import transforms
from tqdm import tqdm

from model.continual_model import CL_MODEL
from model.buffer import BUFFER

class ERMIR(CL_MODEL):
    
    def __init__(self, 
                 nclasses, 
                 buffer_memory_size,
                 buffer_sampling_size, 
                 buffer_batch_size,
                 image_shape, 
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)
        
        self.buffer = BUFFER(buffer_memory_size, image_shape)
        self.buffer_sampling_size = buffer_sampling_size
        self.buffer_batch_size = buffer_batch_size
        self._DEVICE = _DEVICE
    
    def virtual_update(self, inputs, labels):
        
        self.virtual_cl_model = copy.deepcopy(self)
        self.virtual_cl_model.backbone.train()
        self.virtual_cl_model.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self.virtual_cl_model._DEVICE), labels.to(self.virtual_cl_model._DEVICE)
        
        outputs = self.virtual_cl_model.backbone(inputs)
        
        loss = self.virtual_cl_model.loss(outputs, labels)
        loss.backward()
        self.virtual_cl_model.optimizer.step()
        
        self.virtual_cl_model.backbone.eval()
        
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
    
    def mir_sampling(self, sampled_inputs, sampled_labels):
        
        self.backbone.eval()
        self.virtual_cl_model.backbone.eval()
        temp = []
        with torch.no_grad():
            for _idx, (input, label) in enumerate(zip(sampled_inputs, sampled_labels)):
                input, label = input.unsqueeze(0), label.unsqueeze(0)
                input, label = input.to(self._DEVICE), label.to(self._DEVICE)
                output = self.virtual_cl_model.backbone(input)
                loss_virtual = self.virtual_cl_model.loss(output, label)
                output = self.backbone(input)
                loss_real = self.loss(output, label)
                
                diff = loss_virtual.item() - loss_real.item()
                temp.append([_idx, diff])
        temp.sort(key = lambda x:x[1],reverse = True) # 내림차순 정렬, top-k를 뽑아야 함
        temp = temp[:self.buffer_batch_size]
        temp = [i[0] for i in temp]
        self.backbone.train()
        self.virtual_cl_model = None
        
        return sampled_inputs[temp], sampled_labels[temp]
        
        
    
    def buffer_sampling(self):
        
        if self.buffer.num_seen_examples > 0:
            sampled_inputs, sampled_labels = self.buffer.get_data(self.buffer_sampling_size)
            
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

def er_mir_train_example(cfg, train, test):
    
    cl_model = ERMIR(nclasses = cfg.nclasses,
                     buffer_memory_size = cfg.buffer_memory_size,
                     buffer_sampling_size = cfg.buffer_batch_size * 2,
                     buffer_batch_size = cfg.buffer_batch_size,
                     image_shape = cfg.image_shape,
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
                                           ncols = 100):

                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels)
        else:
            for epoch in range(cfg.epoch):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total = len(train_loader),
                                           ncols = 100):

                    cl_model.virtual_update(inputs, labels)
                    sampled_inputs, sampled_labels = cl_model.buffer_sampling()
                    
                    sampled_inputs, sampled_labels = cl_model.mir_sampling(sampled_inputs, sampled_labels)
                    
                    inputs = torch.cat((inputs, sampled_inputs))
                    labels = torch.cat((labels, sampled_labels))
                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs[:cl_model.buffer_batch_size], labels[:cl_model.buffer_batch_size])
        
        for _incremental_time, test_loader in enumerate(val_loader_list[:_incremental_time+1]):
            test_reulsts = cl_model.eval(test_loader, _incremental_time)
            print(test_reulsts)
    
    AVG_ACC = 0
    for _incremental_time, test_loader in enumerate(val_loader_list):
        test_reulsts = cl_model.eval(test_loader, _incremental_time)
        AVG_ACC += test_reulsts[f'Task_{_incremental_time}_EVAL_ACC']
    AVG_ACC /= (cfg.num_increments)
    print(f'\n Average Accuracy = {AVG_ACC}')
    
    return AVG_ACC
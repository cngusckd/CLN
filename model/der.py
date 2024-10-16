import torch
import torch.nn as nn

from tqdm import tqdm


from model.continual_model import CL_MODEL
from model.buffer import DERBUFFER

class DER(CL_MODEL):
    def __init__(self, 
                 nclasses, 
                 buffer_memory_size, 
                 buffer_batch_size, 
                 image_shape, 
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)
        
        self._DEVICE = _DEVICE
        self.buffer = DERBUFFER(buffer_memory_size, image_shape)
        self.alpha = 0.5  # hyperparameter for balancing replay loss
    
    def observe(self, inputs, labels):
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
        outputs = self.backbone(inputs)
        
        # Calculate standard classification loss
        loss = self.loss(outputs, labels)
        
        # Calculate Distillation loss from buffer samples
        if len(self.buffer) > 0:
            buffer_inputs, buffer_labels, buffer_logits = self.buffer.get_data(self.buffer_batch_size)
            buffer_inputs, buffer_labels, buffer_logits = buffer_inputs.to(self._DEVICE), buffer_labels.to(self._DEVICE), buffer_logits.to(self._DEVICE)
            buffer_outputs = self.backbone(buffer_inputs)
            
            distillation_loss = nn.MSELoss()(buffer_outputs, buffer_logits)
            loss += self.alpha * distillation_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def buffer_update(self, inputs, labels, logits):
        for input_data, input_label, input_logits in zip(inputs, labels, logits):
            self.buffer.update(input_data=input_data, input_label=input_label, input_logits=input_logits)
    
    def buffer_sampling(self):
        if len(self.buffer) > 0:
            sampled_inputs, sampled_labels, sampled_logits = self.buffer.get_data(self.buffer_batch_size)
            return sampled_inputs, sampled_labels, sampled_logits
        return None, None, None
    
    def eval(self, val_loader, _incremental_time):
        temp = {}
        
        self.backbone.eval()
        val_loss = 0
        val_acc = 0
        
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
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

class DERPP(CL_MODEL):
    def __init__(self, 
                 nclasses, 
                 buffer_memory_size, 
                 buffer_batch_size, 
                 image_shape, 
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)
        
        self._DEVICE = _DEVICE
        self.buffer = DERBUFFER(buffer_memory_size, image_shape)
        self.alpha = 0.5  # hyperparameter for balancing replay loss
        self.beta = 0.1   # hyperparameter for regularization loss
    
    def observe(self, inputs, labels):
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
        outputs = self.backbone(inputs)
        
        # Calculate standard classification loss
        loss = self.loss(outputs, labels)
        
        # Calculate Distillation loss from buffer samples
        if len(self.buffer) > 0:
            buffer_inputs, buffer_labels, buffer_logits = self.buffer.get_data(self.buffer_batch_size)
            buffer_inputs, buffer_labels, buffer_logits = buffer_inputs.to(self._DEVICE), buffer_labels.to(self._DEVICE), buffer_logits.to(self._DEVICE)
            buffer_outputs = self.backbone(buffer_inputs)
            
            distillation_loss = nn.MSELoss()(buffer_outputs, buffer_logits)
            loss += self.alpha * distillation_loss
            
            # Regularization loss
            regularization_loss = nn.CrossEntropyLoss()(buffer_outputs, buffer_labels)
            loss += self.beta * regularization_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def buffer_update(self, inputs, labels, logits):
        for input_data, input_label, input_logits in zip(inputs, labels, logits):
            self.buffer.update(input_data=input_data, input_label=input_label, input_logits=input_logits)
    
    def buffer_sampling(self):
        if len(self.buffer) > 0:
            sampled_inputs, sampled_labels, sampled_logits = self.buffer.get_data(self.buffer_batch_size)
            return sampled_inputs, sampled_labels, sampled_logits
        return None, None, None
    
    def eval(self, val_loader, _incremental_time):
        temp = {}
        
        self.backbone.eval()
        val_loss = 0
        val_acc = 0
        
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
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

def der_train_example(cfg, train, test):
    cl_model = DER(nclasses = cfg.nclasses,
                   buffer_memory_size = cfg.buffer_memory_size,
                   buffer_batch_size = cfg.buffer_batch_size,
                   image_shape= cfg.image_shape,
                   _DEVICE=torch.device(cfg.device))
    
    val_loader_list = []
    for _idx in range(cfg.num_increments):
        val_loader_list.append(test.get_incremental_loader(_idx))
    
    for _incremental_time in range(cfg.num_increments):
        train_loader = train.get_incremental_loader(_incremental_time)
        
        if _incremental_time == 0:
            for epoch in range(cfg.epoch):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total=len(train_loader),
                                           ncols = 100):
                    with torch.no_grad():
                        cl_model.backbone.eval()
                        outputs = cl_model.backbone(inputs.to(cl_model._DEVICE))
                        cl_model.backbone.train()
                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels, outputs.detach())
        else:
            for epoch in range(cfg.epoch):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total=len(train_loader),
                                           ncols = 100):
                    sampled_inputs, sampled_labels, sampled_logits = cl_model.buffer_sampling()
                    if sampled_inputs is not None:
                        inputs = torch.cat((inputs, sampled_inputs))
                        labels = torch.cat((labels, sampled_labels))
                    
                    outputs = cl_model.backbone(inputs.to(cl_model._DEVICE))
                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels, outputs.detach())
        
        for _incremental_time, test_loader in enumerate(val_loader_list[:_incremental_time + 1]):
            test_results = cl_model.eval(test_loader, _incremental_time)
            print(test_results)
    AVG_ACC = 0
    for _incremental_time, test_loader in enumerate(val_loader_list):
        test_reulsts = cl_model.eval(test_loader, _incremental_time)
        AVG_ACC += test_reulsts[f'Task_{_incremental_time}_EVAL_ACC']
    AVG_ACC /= (cfg.num_increments)
    print(f'\n Average Accuracy = {AVG_ACC}')
    
    return AVG_ACC

def derpp_train_example(cfg, train, test):
    cl_model = DERPP(nclasses = cfg.nclasses,
                     buffer_memory_size = cfg.buffer_memory_size,
                     buffer_batch_size = cfg.buffer_batch_size,
                     image_shape = cfg.image_shape,
                     _DEVICE=torch.device(cfg.device))
    
    val_loader_list = []
    for _idx in range(cfg.num_increments):
        val_loader_list.append(test.get_incremental_loader(_idx))
    
    for _incremental_time in range(cfg.num_increments):
        train_loader = train.get_incremental_loader(_incremental_time)
        
        if _incremental_time == 0:
            for epoch in range(cfg.epoch):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total=len(train_loader),
                                           ncols = 100):
                    outputs = cl_model.backbone(inputs.to(cl_model._DEVICE))
                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels, outputs.detach())
        else:
            for epoch in range(cfg.epoch):
                for inputs, labels in tqdm(train_loader,
                                           desc=f'Task {_incremental_time} Epoch {epoch} Training....',
                                           total=len(train_loader),
                                           ncols = 100):
                    sampled_inputs, sampled_labels, sampled_logits = cl_model.buffer_sampling()
                    if sampled_inputs is not None:
                        inputs = torch.cat((inputs, sampled_inputs))
                        labels = torch.cat((labels, sampled_labels))
                    
                    outputs = cl_model.backbone(inputs.to(cl_model._DEVICE))
                    cl_model.observe(inputs, labels)
                    cl_model.buffer_update(inputs, labels, outputs.detach())
        
        for _incremental_time, test_loader in enumerate(val_loader_list[:_incremental_time + 1]):
            test_results = cl_model.eval(test_loader, _incremental_time)
            print(test_results)
    AVG_ACC = 0
    for _incremental_time, test_loader in enumerate(val_loader_list):
        test_reulsts = cl_model.eval(test_loader, _incremental_time)
        AVG_ACC += test_reulsts[f'Task_{_incremental_time}_EVAL_ACC']
    AVG_ACC /= (cfg.num_increments)
    print(f'\n Average Accuracy = {AVG_ACC}')
    
    return AVG_ACC
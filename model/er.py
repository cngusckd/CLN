import torch

from tqdm import tqdm

from model.buffer import BUFFER
from model.continual_model import CL_MODEL

class ER(CL_MODEL):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.buffer = BUFFER(buffer_memory_size = cfg.buffer_memory_size,
                             image_shape = cfg.image_shape,
                             buffer_extraction = cfg.buffer_extraction,
                             buffer_extraction_size = cfg.buffer_extraction_size,
                             buffer_storage = cfg.buffer_storage,
                             buffer_storage_size = cfg.buffer_storage_size)
        
        self.cfg = cfg
    
    def train_task(self, train_loader):
        
        if self.current_task_index == 0:
            # first task
            
            for _epoch in tqdm(range(self.cfg.epoch),
                               desc = f'Task {self.current_task_index} training... / epoch : {_epoch}',
                               ncols = 120):
                
                for inputs, labels in train_loader:
                    
                    self.observe(inputs, labels)
                    self.buffer.store(inputs, labels)
            
            self.current_task_index += 1
        
        else:
            
            for _epoch in tqdm(range(self.cfg.epoch),
                               desc = f'Task {self.current_task_index} training... / epoch : {_epoch}',
                               ncols = 120):
                
                for inputs, labels in train_loader:
                    
                    sampled_inputs, sampled_labels = self.buffer.extract()
                    
                    self.er_observe(inputs, labels, sampled_inputs, sampled_labels)
                    self.buffer.store(inputs, labels)
            
            self.current_task_index += 1
    
    def observe(self, inputs, labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def er_observe(self, inputs, labels, sampled_inputs, sampled_labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = torch.cat((inputs, sampled_inputs)), torch.cat((labels, sampled_labels))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    
    def eval(self, val_loader, _incremental_time):
    
        temp = {}
        
        self.backbone.eval()
        val_loss = 0
        val_acc = 0
        
        for inputs, labels in val_loader:
            
            inputs, labels, = inputs.to(self.device), labels.to(self.device)
            
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

def er_train_example(cfg, train, test):
    
    cl_model = ER(nclasses = cfg.nclasses,
                  buffer_memory_size = cfg.buffer_memory_size,
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
                    
                    sampled_inputs, sampled_labels = cl_model.buffer_sampling()
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
         
'''
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

def er_train_example(cfg, train, test):
    
    cl_model = ER(nclasses = cfg.nclasses,
                  buffer_memory_size = cfg.buffer_memory_size,
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
                    
                    sampled_inputs, sampled_labels = cl_model.buffer_sampling()
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
'''
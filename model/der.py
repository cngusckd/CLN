import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from tqdm import tqdm


from model.continual_model import CL_MODEL
from model.buffer import DarkExperienceBuffer

class DER(CL_MODEL):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.buffer = DarkExperienceBuffer(cfg)
    
    def train_task(self, train_loader):
        
        if self.current_task_index == 0 :
            # frist task
            
            pbar = tqdm(range(self.cfg.epoch), ncols = 120)
            for _epoch in pbar:
                pbar.set_description(f'Task {self.current_task_index} training... / epoch : {_epoch}')
                
                for inputs, labels in train_loader:
                    
                    logits = self.observe(inputs, labels) # return output logit
                    self.store(inputs, labels, logits)
                    
            self.current_task_index += 1
            
        else:
            #incremental task
            
            pbar = tqdm(range(self.cfg.epoch), ncols = 120)
            for _epoch in pbar:
                pbar.set_description(f'Task {self.current_task_index} training... / epoch : {_epoch}')
                
                for inputs, labels in train_loader:
                    
                    if self.cfg.buffer_extraction == 'mir':
                        self.virtual_update(inputs, labels)
                        sampled_inputs, sampled_labels, sampled_logits, _index_list = self.mir_sampling()
                    else:
                        sampled_inputs, sampled_labels, sampled_logits, _index_list = self.extract()
                    
                    logits = self.joint_observe(inputs, labels, sampled_inputs, sampled_labels, sampled_logits)
                    
                    if self.cfg.buffer_storage == 'gss':
                        self.gss_store(inputs, labels, logits, sampled_inputs, sampled_labels, _index_list)
                    else:
                        self.store(inputs, labels, logits)
                        
            self.current_task_index += 1
            
    def observe(self, inputs, labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return outputs.detach()
    
    def store(self, inputs, labels, logits):
        
        # store with resorvoir sampling
        
        for input, label, logit in zip(inputs, labels, logits):
            self.buffer.store(input_data = input,
                              input_label = label,
                              input_logits = logit)
    
    def gss_store(self, inputs, labels, logits, sampled_inputs, sampled_labels, index_list):
        
        
        
        # it is different with paper, calculate gradients one by one instead of across the batch
        # see algorithm2 greedy sample selection in the paper
        # modified to just replace directly if consine sim < 0
        for _index, (input, label) in enumerate(zip(inputs, labels)):
            new_gradient = self.compute_gradients(inputs[_index], labels[_index])
            gradient = self.compute_gradients(sampled_inputs[_index], sampled_labels[_index])
            cosine_sim = self.calculate_cosine_similarity(grads1 = new_gradient,
                                                          grads2 = gradient)
            if cosine_sim < 0:
                self.buffer.index_store(input_data = inputs[_index],
                                        input_label = labels[_index],
                                        logit = logits[_index],
                                        index = index_list[_index])
    
    def compute_gradients(self, inputs, labels):
        
        self.backbone.eval()
        self.optimizer.zero_grad()
        outputs = self.backbone(inputs.unsqueeze(0).to(self.device))
        loss = self.loss(outputs, labels.unsqueeze(0).to(self.device))
        loss.backward()
        grads = self.get_grads().clone().detach()
        self.backbone.zero_grad()
        self.backbone.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads   
    
    def calculate_cosine_similarity(self, grads1, grads2):
        # L2 normalization
        grads1_norm = F.normalize(grads1, p=2, dim=1)
        grads2_norm = F.normalize(grads2, p=2, dim=1)
    
        # calculate cosine similarity with torch.mm
        cosine_similarity = torch.mm(grads1_norm, grads2_norm.T)
    
        return cosine_similarity
       
    def extract(self):
        
        return self.buffer.extract()
    
    def virtual_update(self, inputs, labels):
        # need for mir sampling
        
        self.virtual_cl_model = copy.deepcopy(self)
        self.virtual_cl_model.backbone.train()
        self.virtual_cl_model.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self.virtual_cl_model.device), labels.to(self.virtual_cl_model.device)
        
        outputs = self.virtual_cl_model.backbone(inputs)
        
        loss = self.virtual_cl_model.loss(outputs, labels)
        loss.backward()
        self.virtual_cl_model.optimizer.step()
        
        self.virtual_cl_model.backbone.eval()
    
    def mir_sampling(self):
        # code for mir sampling
        self.backbone.eval()
        self.virtual_cl_model.backbone.eval()
        temp = []
        
        sampled_inputs, sampled_labels, sampled_logits, index_list = self.buffer.extract()
        
        with torch.no_grad():
            for _idx, (input, label) in enumerate(zip(sampled_inputs, sampled_labels)):
                input, label = input.unsqueeze(0), label.unsqueeze(0)
                input, label = input.to(self.device), label.to(self.device)
                output = self.virtual_cl_model.backbone(input)
                loss_virtual = self.virtual_cl_model.loss(output, label)
                output = self.backbone(input)
                loss_real = self.loss(output, label)
                
                diff = loss_virtual.item() - loss_real.item()
                temp.append([_idx, diff])
        
        temp.sort(key = lambda x:x[1],reverse = True)
        # descending order, need to extract top-k(cfg.buffer_extraction_size)
        temp = temp[:self.cfg.buffer_extraction_size]
        # buffer_extraction_size is defined as batch_size for joint training
        temp = [i[0] for i in temp]
        self.backbone.train()
        self.virtual_cl_model = None
        
        return sampled_inputs[temp], sampled_labels[temp], sampled_logits[temp], temp
        
    
    def joint_observe(self, inputs, labels, sampled_inputs, sampled_labels, sampled_logits):
        
        self.optimizer.zero_grad()
        
        # loss for current stream
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.backbone(inputs)
        loss = self.loss(outputs, labels)
        
        sampled_inputs, sampled_labels, sampled_logits = sampled_inputs.to(self.device), sampled_labels.to(self.device), sampled_logits.to(self.device)
        sample_outputs = self.backbone(sampled_inputs)
        distillation_loss = nn.MSELoss()(sample_outputs, sampled_logits)
        
        loss += self.cfg.alpha * distillation_loss
        
        if self.cfg.model == 'der++':
            regularization_loss = nn.CrossEntropyLoss()(sample_outputs, sampled_labels)
            loss += self.cfg.beta * regularization_loss
        
        
        loss.backward()
        self.optimizer.step()
        
        return outputs.detach()

    def eval_task(self, val_loader, task_index):
    
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
        temp[f'Task_{task_index}_EVAL_ACC'] = val_acc.item()
        temp[f'Task_{task_index}_EVAL_LOSS'] = val_loss
        self.backbone.train()
        
        return temp
    
# class DER(CL_MODEL):
#     def __init__(self, 
#                  nclasses, 
#                  buffer_memory_size, 
#                  buffer_batch_size, 
#                  image_shape, 
#                  _DEVICE):
#         super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)
        
#         self._DEVICE = _DEVICE
#         self.buffer = DERBUFFER(buffer_memory_size, image_shape)
#         self.alpha = 0.5  # hyperparameter for balancing replay loss
    
#     def observe(self, inputs, labels):
#         self.optimizer.zero_grad()
        
#         inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
#         outputs = self.backbone(inputs)
        
#         # Calculate standard classification loss
#         loss = self.loss(outputs, labels)
        
#         # Calculate Distillation loss from buffer samples
#         if len(self.buffer) > 0:
#             buffer_inputs, buffer_labels, buffer_logits = self.buffer.get_data(self.buffer_batch_size)
#             buffer_inputs, buffer_labels, buffer_logits = buffer_inputs.to(self._DEVICE), buffer_labels.to(self._DEVICE), buffer_logits.to(self._DEVICE)
#             buffer_outputs = self.backbone(buffer_inputs)
            
#             distillation_loss = nn.MSELoss()(buffer_outputs, buffer_logits)
#             loss += self.alpha * distillation_loss
        
#         loss.backward()
#         self.optimizer.step()
        
#         return loss.item()
    
#     def buffer_update(self, inputs, labels, logits):
#         for input_data, input_label, input_logits in zip(inputs, labels, logits):
#             self.buffer.update(input_data=input_data, input_label=input_label, input_logits=input_logits)
    
#     def buffer_sampling(self):
#         if len(self.buffer) > 0:
#             sampled_inputs, sampled_labels, sampled_logits = self.buffer.get_data(self.buffer_batch_size)
#             return sampled_inputs, sampled_labels, sampled_logits
#         return None, None, None
    
#     def eval(self, val_loader, _incremental_time):
#         temp = {}
        
#         self.backbone.eval()
#         val_loss = 0
#         val_acc = 0
        
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
#             outputs = self.backbone(inputs)
#             loss = self.loss(outputs, labels)
            
#             val_loss += (loss.item() / len(inputs))
            
#             pred = outputs.argmax(dim=1)
#             val_acc += (pred == labels).float().sum()
        
#         val_acc /= len(val_loader.dataset)
#         torch.cuda.empty_cache()
#         temp[f'Task_{_incremental_time}_EVAL_ACC'] = val_acc.item()
#         temp[f'Task_{_incremental_time}_EVAL_LOSS'] = val_loss
#         self.backbone.train()
        
#         return temp

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
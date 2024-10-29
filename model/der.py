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
        
    def Logger(self):
        
        raise NotImplementedError
    
    def train_task(self, train_loader):
        
        if self.current_task_index == 0 :
            # frist task
            
            pbar = tqdm(range(self.cfg.epoch), ncols = 120)
            for _epoch in pbar:
                pbar.set_description(f'Task {self.current_task_index} training... / epoch : {_epoch}')
                
                for inputs, labels in train_loader:
                    
                    loss, logits = self.observe(inputs, labels) # return output logit
                    self.store(inputs, labels, logits)
                    if self.cfg.wandb:
                        self.wandb_train_logger(loss)
                    
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
                    
                    loss, logits = self.joint_observe(inputs, labels, sampled_inputs, sampled_labels, sampled_logits)
                    
                    if self.cfg.buffer_storage == 'gss':
                        self.gss_store(inputs, labels, logits, sampled_inputs, sampled_labels, _index_list)
                    else:
                        self.store(inputs, labels, logits)
                        
                    if self.cfg.wandb:
                        self.wandb_train_logger(loss)
                        
            self.current_task_index += 1
            
    def observe(self, inputs, labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), outputs.detach()
    
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
        
        return loss.item(), outputs.detach()
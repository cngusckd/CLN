import torch
import copy
import torch.nn.functional as F

from tqdm import tqdm

from model.buffer import DefaultBuffer
from model.continual_model import CL_MODEL

class ER(CL_MODEL):
    
    def __init__(self, cfg):
        super().__init__(cfg)
    
        # self.cfg = cfg

    def train_task(self, train_loader):
        
        if self.current_task_index == 0:
            # first task
            
            pbar = tqdm(range(self.cfg.epoch), ncols = 120)
            for _epoch in pbar:
                
                for inputs, labels in train_loader:
                    
                    self.observe(inputs, labels)
                    self.store(inputs, labels)
            
                pbar.set_description(f'Task {self.current_task_index} training... / epoch : {_epoch}')
            self.current_task_index += 1
        
        else:
            # incremental task
            pbar = tqdm(range(self.cfg.epoch), ncols = 120)
            for _epoch in pbar:
                
                for inputs, labels in train_loader:
                    
                    
                    if self.cfg.buffer_extraction == 'mir':
                        self.virtual_update(inputs, labels)
                        sampled_inputs, sampled_labels, _index_list = self.mir_sampling()
                    else:
                        sampled_inputs, sampled_labels, _index_list = self.extract()
                    
                    self.joint_observe(inputs, labels, sampled_inputs, sampled_labels)
                    
                    if self.cfg.buffer_storage == 'gss':
                        self.gss_store(inputs, labels, sampled_inputs, sampled_labels, _index_list)
                    else:
                        self.store(inputs, labels)
                    
                pbar.set_description(f'Task {self.current_task_index} training... / epoch : {_epoch}')
            self.current_task_index += 1
    
    def observe(self, inputs, labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def joint_observe(self, inputs, labels, sampled_inputs, sampled_labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = torch.cat((inputs, sampled_inputs)), torch.cat((labels, sampled_labels))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def extract(self):

        # code for random sampling
        return self.buffer.extract()
    
    def mir_sampling(self):
            # code for mir sampling
            self.backbone.eval()
            self.virtual_cl_model.backbone.eval()
            temp = []
            
            sampled_inputs, sampled_labels, index_list = self.buffer.extract()
            
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
            
            return sampled_inputs[temp], sampled_labels[temp], temp
    
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
    
    def store(self, inputs, labels):
        # store with resorvoir sampling
        for input_data, input_label in zip(inputs, labels):
            self.buffer.store(input_data = input_data,
                                input_label = input_label)
    
    def gss_store(self, inputs, labels, sampled_inputs, sampled_labels, index_list):
        
        # batch 통째로 gradient를 계산하는것이 아닌 하나 계산
        # 논문의 Algorithm2 Greedy Sample Selection 참고
        # consine sim < 0 일경우 그냥 바로 바꾸도록 수정함
        change_index = [] # buffer의 index
        current_index = [] # current stream의 index
        for _index, (input, label, sampled_input, sampled_label) in enumerate(zip(inputs, labels, sampled_inputs, sampled_labels)):
            new_gradient = self.compute_gradients(input, label)
            gradient = self.compute_gradients(sampled_input, sampled_label)
            cosine_sim = self.calculate_cosine_similarity(grads1 = new_gradient,
                                                          grads2 = gradient)
            if cosine_sim < 0:
                change_index.append(index_list[_index]) # 바꾸어야할 index 확인
                current_index.append(_index)
                
        if len(change_index) >= 1:
            for _index, (input_data, input_label) in enumerate(zip(inputs, labels)):
                self.buffer.index_store(input_data = input_data,
                                        input_label = input_label,
                                        index = index_list[_index])
        else:
            pass
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
        # 코사인 유사도를 계산하기 전에 L2 정규화를 수행합니다.
        grads1_norm = F.normalize(grads1, p=2, dim=1)
        grads2_norm = F.normalize(grads2, p=2, dim=1)
    
        # 코사인 유사도 계산 (벡터 곱)
        cosine_similarity = torch.mm(grads1_norm, grads2_norm.T)
    
        return cosine_similarity   
    
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
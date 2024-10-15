import torch
import torch.nn.functional as F

from torchvision import transforms
from tqdm import tqdm

from model.continual_model import CL_MODEL
from model.buffer import GSSBuffer

class ERGSS(CL_MODEL):
    def __init__(self, 
                 nclasses, 
                 buffer_memory_size, 
                 buffer_batch_size, 
                 image_shape, 
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)
        
        self.buffer = GSSBuffer(buffer_memory_size = buffer_memory_size,
                                image_shape = image_shape)
        self._DEVICE = _DEVICE
    
    def observe(self, inputs, labels):
        # Task 0에 대한 학습
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
    
    def gss_update(self, inputs, labels, index_list):
        
        for _index, (input_data, input_label) in enumerate(zip(inputs, labels)):
            self.buffer.gss_update(input_data = input_data,
                                   input_label = input_label,
                                   index = index_list[_index])
    
    def calculate_cosine_similarity(self, grads1, grads2):
        # 코사인 유사도를 계산하기 전에 L2 정규화를 수행합니다.
        grads1_norm = F.normalize(grads1, p=2, dim=1)
        grads2_norm = F.normalize(grads2, p=2, dim=1)
    
        # 코사인 유사도 계산 (벡터 곱)
        cosine_similarity = torch.mm(grads1_norm, grads2_norm.T)
    
        return cosine_similarity
    
    def compute_gradients(self, inputs, labels):
        self.backbone.eval()
        self.optimizer.zero_grad()
        outputs = self.backbone(inputs.unsqueeze(0).to(self._DEVICE))
        loss = self.loss(outputs, labels.unsqueeze(0).to(self._DEVICE))
        loss.backward()
        grads = self.get_grads().clone().detach()
        self.backbone.zero_grad()
        self.backbone.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads
    
    def buffer_sampling(self):
        
        if self.buffer.num_seen_examples > 0:
            sampled_inputs, sampled_labels, index_list = self.buffer.get_data(self.buffer_batch_size)
            
        return sampled_inputs, sampled_labels, index_list
    
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

def er_gss_train_example(cfg, train, test):
    
    cl_model = ERGSS(nclasses = cfg.nclasses,
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
                    
                    # batch 통째로 gradient를 계산하는것이 아닌 하나 계산
                    # 논문의 Algorithm2 Greedy Sample Selection 참고
                    # consine sim < 0 일경우 그냥 바로 바꾸도록 수정함
                    sampled_inputs, sampled_labels, index_list = cl_model.buffer_sampling()
                    change_index = [] # buffer의 index
                    current_index = [] # current stream의 index
                    for _index, (input, label, sampled_input, sampled_label) in enumerate(zip(inputs, labels, sampled_inputs, sampled_labels)):
                        new_gradient = cl_model.compute_gradients(input, label)
                        gradient = cl_model.compute_gradients(sampled_input, sampled_label)
                        cosine_sim = cl_model.calculate_cosine_similarity(grads1 = new_gradient,
                                                                          grads2 = gradient)
                        if cosine_sim < 0:
                            change_index.append(index_list[_index]) # 바꾸어야할 index 확인
                            current_index.append(_index)
                    
                    
                    if len(change_index) > 1:
                        cl_model.gss_update(inputs[current_index], labels[current_index], change_index)
                    elif len(change_index) == 1:
                        _index = current_index[0]
                        cl_model.buffer.gss_update(input[_index], labels[_index], _index)
                    inputs = torch.cat((inputs, sampled_inputs))
                    labels = torch.cat((labels, sampled_labels))
                    cl_model.observe(inputs, labels)
                    # print(cl_model.buffer.labels.shape)
                    # print(f'buffer dsitribution {torch.bincount(cl_model.buffer.labels.squeeze())}')
                    
        
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
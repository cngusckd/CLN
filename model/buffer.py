import torch
import numpy as np
import random

class DefaultBuffer:
    
    def __init__(self, cfg):
        
        self.num_seen_examples = 0
        self.buffer_memory_size = cfg.buffer_memory_size
        self.image_shape = cfg.image_shape
        # init buffer
        
        self.buffer_extraction = cfg.buffer_extraction
        # exemplar extraction method & size
        
        if self.buffer_extraction == 'mir':
            self.buffer_extraction_size = cfg.buffer_extraction_size * 2
            # for mir extraction strategy, extract twice as many exemplars
            # in can be modified by user
        elif self.buffer_extraction == 'random':
            self.buffer_extraction_size = cfg.buffer_extraction_size
        else:
            raise NotImplementedError
        
        self.buffer_storage = cfg.buffer_storage
        self.buffer_storage_size = cfg.buffer_storage_size
        # exemplar storage method & size
        
        self.examples = torch.zeros((self.buffer_memory_size, 3, self.image_shape[0], self.image_shape[1]), dtype = torch.float32)
        self.labels = torch.zeros((self.buffer_memory_size, 1), dtype = torch.int64)
        
    
    def __len__(self):
        return min(self.num_seen_examples, self.buffer_memory_size)

    def extract(self): # method for random extraction
            
        temp_list = [i for i in range(min(self.num_seen_examples, self.buffer_memory_size))]
        index_list = random.sample(temp_list, min(self.num_seen_examples, len(temp_list)))
        
        _return_example_list = []
        _return_label_list = []
        for _idx in index_list:
            _return_example_list.append(self.examples[_idx])
            _return_label_list.append(self.labels[_idx])
        try:
            _return_example_list = torch.stack(_return_example_list)
            _return_label_list = torch.stack(_return_label_list)
            _return_label_list = _return_label_list.squeeze()
        except:
            print(_return_example_list)
            print(_return_label_list)
            exit()
        
        return _return_example_list, _return_label_list, index_list
    
    
    def store(self, input_data, input_label):
        
        self.num_seen_examples += 1

        if self.num_seen_examples <= self.buffer_memory_size:
            # if buffer is not full, add exemplar
            self.examples[self.num_seen_examples - 1] = input_data
            self.labels[self.num_seen_examples - 1] = input_label
        else:
            # if buffer is full, store exemplar based on reservoir sampling
            rand_index = np.random.randint(0, self.num_seen_examples)
            if rand_index < self.buffer_memory_size:
                self.examples[rand_index] = input_data
                self.labels[rand_index] = input_label       
    
    def index_store(self, input_data, input_label, index):
        
        if self.num_seen_examples < self.buffer_memory_size:
        # 버퍼에 여유 공간이 있는 경우 샘플 추가
            change_index = self.num_seen_examples    
        else:
            # 버퍼가 가득 찬 경우, 전달받은 index로 수정
            change_index = index
        
        self.examples[change_index] = input_data
        self.labels[change_index] = input_label
        self.num_seen_examples += 1


        
        
        
        

class BUFFER:
    '''
    메모리 버퍼
    '''
    
    def __init__(self, buffer_memory_size, image_shape) -> None:
        # 버퍼 사이즈에 대해 버퍼 초기화 진행
        self.buffer_memory_size = buffer_memory_size
        self.image_shape = image_shape
        self.examples = torch.zeros((self.buffer_memory_size, 3, self.image_shape[0], self.image_shape[1]), dtype = torch.float32)
        self.labels = torch.zeros((self.buffer_memory_size, 1), dtype = torch.int64)
        
        self.num_seen_examples = 0
        
    def __len__(self):
        return min(self.num_seen_examples, self.buffer_memory_size)
    
    def update(self, input_data, input_label):
        
        self.num_seen_examples += 1

        if self.num_seen_examples < self.buffer_memory_size:
            # 버퍼가 꽉 차 있지 않은 경우, 그냥 추가함
            self.examples[self.num_seen_examples] = input_data
            self.labels[self.num_seen_examples] = input_label
        
        else:
            # 버퍼가 차 차있을 경우 reservoir update 진행
            rand_index = np.random.randint(0, self.num_seen_examples + 1)
            
            if rand_index < self.buffer_memory_size:
                self.examples[rand_index] = input_data
                self.labels[rand_index] = input_label
    
    def store(self, input_data, input_label):
        
        self.num_seen_examples += 1

        if self.num_seen_examples <= self.buffer_memory_size:
            # if buffer is not full, add exemplar
            self.examples[self.num_seen_examples - 1] = input_data
            self.labels[self.num_seen_examples - 1] = input_label
        else:
            # if buffer is full, store exemplar based on reservoir sampling
            rand_index = np.random.randint(0, self.num_seen_examples)
            if rand_index < self.buffer_memory_size:
                self.examples[rand_index] = input_data
                self.labels[rand_index] = input_label
    
    def get_data(self, data_num): # 난수 추출로 일단 구현함
        
        temp_list = [i for i in range(min(self.num_seen_examples, self.buffer_memory_size))]
        index_list = random.sample(temp_list, min(data_num, len(temp_list)))
        
        _return_example_list = []
        _return_label_list = []
        for _idx in index_list:
            _return_example_list.append(self.examples[_idx])
            _return_label_list.append(self.labels[_idx])
        
        _return_example_list = torch.stack(_return_example_list)
        _return_label_list = torch.stack(_return_label_list)
        _return_label_list = _return_label_list.squeeze()

        return _return_example_list, _return_label_list

class DERBUFFER:
    '''
    DER 메모리 버퍼
    '''
    
    def __init__(self, buffer_memory_size, image_shape) -> None:
        # 버퍼 사이즈에 대해 버퍼 초기화 진행
        self.buffer_memory_size = buffer_memory_size
        self.image_shape = image_shape
        self.examples = torch.zeros((self.buffer_memory_size, 3, self.image_shape[0], self.image_shape[1]), dtype=torch.float32)
        self.labels = torch.zeros((self.buffer_memory_size, 1), dtype=torch.int64)
        self.logits = torch.zeros((self.buffer_memory_size, 10), dtype=torch.float32)  # Store output logits for distillation
        
        self.num_seen_examples = 0
    
    def __len__(self):
        return min(self.num_seen_examples, self.buffer_memory_size)
    
    def update(self, input_data, input_label, input_logits):
        self.num_seen_examples += 1

        if self.num_seen_examples <= self.buffer_memory_size:
            # 버퍼가 꽉 차 있지 않은 경우, 그냥 추가함
            self.examples[self.num_seen_examples - 1] = input_data
            self.labels[self.num_seen_examples - 1] = input_label
            self.logits[self.num_seen_examples - 1] = input_logits
        else:
            # 버퍼가 차 있을 경우 reservoir update 진행
            rand_index = np.random.randint(0, self.num_seen_examples)
            if rand_index < self.buffer_memory_size:
                self.examples[rand_index] = input_data
                self.labels[rand_index] = input_label
                self.logits[rand_index] = input_logits
    
    def get_data(self, data_num):  # 난수 추출로 일단 구현함
        temp_list = [i for i in range(min(self.num_seen_examples, self.buffer_memory_size))]
        index_list = random.sample(temp_list, min(data_num, len(temp_list)))
        
        _return_example_list = []
        _return_label_list = []
        _return_logits_list = []
        for _idx in index_list:
            _return_example_list.append(self.examples[_idx])
            _return_label_list.append(self.labels[_idx])
            _return_logits_list.append(self.logits[_idx])
        
        _return_example_list = torch.stack(_return_example_list)
        _return_label_list = torch.stack(_return_label_list)
        _return_label_list = _return_label_list.squeeze()
        _return_logits_list = torch.stack(_return_logits_list)

        return _return_example_list, _return_label_list, _return_logits_list
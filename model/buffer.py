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
        
        temp_list = np.arange(min(self.num_seen_examples, self.buffer_memory_size))
        index_list = np.random.choice(temp_list, min(self.num_seen_examples, len(temp_list)), replace=False)
        
        _return_example_list = self.examples[index_list]
        _return_label_list = self.labels[index_list]
        _return_label_list = _return_label_list.squeeze()
        
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
            # if buffer is not full, add exemplar
            change_index = self.num_seen_examples    
        else:
            # if buffer is full, store exemplar based on index
            change_index = index
        
        self.examples[change_index] = input_data
        self.labels[change_index] = input_label
        self.num_seen_examples += 1

class DarkExperienceBuffer:
    
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
        self.logits = torch.zeros((self.buffer_memory_size, cfg.nclasses), dtype = torch.float32)
    
    def __len__(self):
        return min(self.num_seen_examples, self.buffer_memory_size)
       
    def extract(self):
        
        temp_list = np.arange(min(self.num_seen_examples, self.buffer_memory_size))
        index_list = np.random.choice(temp_list, min(self.num_seen_examples, len(temp_list)), replace=False)
        
        _return_example_list = self.examples[index_list]
        _return_label_list = self.labels[index_list]
        _return_label_list = _return_label_list.squeeze()
        _return_logits_list = self.logits[index_list]

        return _return_example_list, _return_label_list, _return_logits_list,index_list

    def store(self, input_data, input_label, input_logits):
        self.num_seen_examples += 1

        if self.num_seen_examples <= self.buffer_memory_size:
            # if buffer is not full, add exemplar
            self.examples[self.num_seen_examples - 1] = input_data
            self.labels[self.num_seen_examples - 1] = input_label
            self.logits[self.num_seen_examples - 1] = input_logits
        else:
            # if buffer is full, store exemplar based on reservoir sampling
            rand_index = np.random.randint(0, self.num_seen_examples)
            if rand_index < self.buffer_memory_size:
                self.examples[rand_index] = input_data
                self.labels[rand_index] = input_label
                self.logits[rand_index] = input_logits
    
    def index_store(self, input_data, input_label, logit, index):
        
        if self.num_seen_examples < self.buffer_memory_size:
            # if buffer is not full, add exemplar
            change_index = self.num_seen_examples    
        else:
            # if buffer is full, store exemplar based on reservoir sampling
            change_index = index
        
        self.examples[change_index] = input_data
        self.labels[change_index] = input_label
        self.logits[change_index] = logit
        self.num_seen_examples += 1
import torch
import numpy as np
import random

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

       
if __name__ == '__main__':
    # Buufer TEST
    
    buffer = BUFFER(buffer_memory_size= 50, image_shape=(64, 64))
    
    for k in range(100):
        test_input = torch.rand((64,64))
        
        test_label = torch.randint(low = 0, high = 10, size = [1])
        
        buffer.update(test_input, test_label)
    
    sampled_input, sampled_labels = buffer.get_data(20)
    print(sampled_input)
    print(sampled_labels)
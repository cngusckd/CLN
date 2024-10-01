import torch

from model.continual_model import CL_MODEL

class ER(CL_MODEL):
    
    def __init__(self, 
                 nclasses, 
                 buffer_memory_size, 
                 buffer_batch_size, 
                 input_size, 
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, input_size, _DEVICE)
        
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
        
        return temp
        
        
    
if __name__=='__main__':
    
    continual_model = ER(10, 100, 10, (64, 64), _DEVICE = torch.device('cuda'))
    
    
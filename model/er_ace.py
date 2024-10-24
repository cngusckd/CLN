import torch

from model.er import ER


class ER_ACE(ER):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def joint_observe(self, inputs, labels, sampled_inputs, sampled_labels):
        
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        # incoming batch masking
        present = labels.unique()
        mask = torch.zeros_like(outputs)
        mask[:, present] = 1
        
        outputs = outputs.masked_fill(mask == 0, -1e9)
        
        new_loss = self.loss(outputs, labels)
        
        # data seperate
        inputs, labels = sampled_inputs.to(self.device), sampled_labels.to(self.device)
        
        outputs = self.backbone(inputs)
        
        buffer_loss = self.loss(outputs, labels)
        
        loss = new_loss + buffer_loss
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
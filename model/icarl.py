import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from model.continual_model import CL_MODEL
from model.buffer import iCaRLBUFFER

class iCaRL(CL_MODEL):
    def __init__(self,
                 nclasses,
                 buffer_memory_size,
                 buffer_batch_size,
                 image_shape,
                 _DEVICE):
        super().__init__(nclasses, buffer_memory_size, buffer_batch_size, image_shape, _DEVICE)
        
        self.buffer = iCaRLBUFFER(buffer_memory_size = buffer_memory_size,
                                  image_shape = image_shape)
        self.exemplars_mean = []
    
    def train_with_data_loader(self, data_loader):
        self.backbone.train()
        for inputs, labels in data_loader:
            self.observe(inputs, labels)
            self.update_exemplars_mean()
        
    def extract_features(self, x):
        return self.backbone(x, returnt='features')
    
    def classify(self, x):
        features = self.extract_features(x)
        features = nn.functional.normalize(features, p=2, dim=1)
        preds = []
        for feature in features:
            distances = [torch.dist(feature, torch.tensor(exemplar_mean).to(self._DEVICE)) for exemplar_mean in self.exemplars_mean]
            preds.append(np.argmin(distances))
        return preds
    
    def update_exemplars_mean(self):
        self.exemplars_mean = []
        with torch.no_grad():
            for exemplar_set in self.buffer.get_memory():
                features = [self.extract_features(img.to(self._DEVICE).unsqueeze(0)).cpu().numpy().flatten() for img, _ in exemplar_set]
                features = np.array(features)
                exemplar_mean = np.mean(features, axis=0)
                self.exemplars_mean.append(exemplar_mean / np.linalg.norm(exemplar_mean))
    
    def observe(self, inputs, labels):
        self.optimizer.zero_grad()
        
        inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
        outputs = self.backbone(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval(self, data_loader, task_num):
        self.backbone.eval()
        total = 0
        correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self._DEVICE), labels.to(self._DEVICE)
                outputs = self.classify(inputs)
                total += labels.size(0)
                correct += (outputs == labels.cpu().numpy()).sum().item()
        
        accuracy = correct / total
        return {f'Task_{task_num}_EVAL_ACC': accuracy * 100, f'Task_{task_num}_EVAL_LOSS': val_loss}

def icarl_train_example(cfg, train, test):
    cl_model = iCaRL(nclasses=cfg.nclasses,
                     buffer_memory_size=cfg.buffer_memory_size,
                     buffer_batch_size=cfg.buffer_batch_size,
                     image_shape=(28, 28),
                     _DEVICE=torch.device(cfg.device))
    
    val_loader_list = []
    for _idx in range(cfg.num_increments):
        val_loader_list.append(test.get_incremental_loader(_idx))
    
    for _incremental_time in range(cfg.num_increments):
        train_loader = train.get_incremental_loader(_incremental_time)
        
        cl_model.train_with_data_loader(train_loader)
        
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
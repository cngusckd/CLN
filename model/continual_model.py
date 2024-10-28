import wandb
import torch
import torch.nn as nn

from model.resnet import resnet18
from model.buffer import DefaultBuffer

from sklearn.metrics import confusion_matrix, roc_auc_score

class CL_MODEL(nn.Module):
    
    def __init__(self,
                 cfg
                 ):
        super().__init__()
        
        
        self.cfg = cfg
        
        self.device = cfg.device
        # device for training
        
        self.backbone = resnet18(nclasses = cfg.nclasses, nf = cfg.image_shape[0]).to(self.device)
        # backbone networks, now only support resnet18
        # you can modify to resnet34,50,101,121 from model/resnet.py
        # nf : input size(transformed input size)
        
        self.optimizer = self.get_optimizer()
        self.loss = self.get_loss_func()
        # optmizer & criteria
        
        self.current_task_index = 0
        # check current task_index for continual learning
        
        self.buffer = DefaultBuffer(cfg)
        # Buffer for continual learning
        
        self.wandb = wandb
    
    def wandb_train_logger(self, loss):
        
        # Calculate buffer distribution in buffer
        class_counts = torch.bincount(self.buffer.labels[:self.buffer.__len__()].squeeze(), minlength = self.cfg.nclasses)
        class_distribution = {f'class_{i}': class_counts[i].item() for i in range(self.cfg.nclasses)}
        
        # Log training metrics with wandb
        self.wandb.log({
            f'Task_{self.current_task_index}_TRAIN_buffer_distribution' : class_distribution, # buffer distribution of iteration
            f'Task_{self.current_task_index}_TRAIN_loss': loss, # train loss of iteration
        })
        
        return None
    
    def wandb_eval_logger(self, val_acc, val_loss, auroc, conf_matrix, task_idx):
        
        # Calculate buffer distribution in buffer
        class_counts = torch.bincount(self.buffer.labels[:self.buffer.__len__()].squeeze(), minlength = self.cfg.nclasses)
        class_distribution = {f'class_{i}': class_counts[i].item() for i in range(self.cfg.nclasses)}

        # Log evaluation metrics with wandb
        self.wandb.log({
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_buffer_distribution' : class_distribution, # buffer distribution of task_idx with current_task_index
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_acc': val_acc, # accuracy of task_idx with current_task_index
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_loss': val_loss, # loss of task_idx with current_task_index
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_auroc': auroc, # auroc of task_idx with current_task_index
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_conf_matrix': conf_matrix # confusion matrix of task_idx with current_task_index
        })
        
        return None
    
    def get_parameters(self):
        
        return self.backbone.parameters()
    
    def get_optimizer(self): # default settings
        
        if self.cfg.optim == 'sgd':
            return torch.optim.SGD(params = self.get_parameters(),
                                   lr = self.cfg.lr,
                                   momentum = self.cfg.momentum)
        else:
            raise NotImplementedError
        
    def get_loss_func(self):
        
        return torch.nn.CrossEntropyLoss()
    
    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.

        Returns:
            gradients tensor
        """
        grads = []
        for pp in list(self.backbone.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    
    def train_task():
        
        raise NotImplementedError
    
    def eval_task(self, val_loader_list):
        self.backbone.eval()
        val_loss = 0
        val_acc = 0
        all_preds = []
        all_labels = []
        all_probs = []  # Added: List to store softmax probabilities for all batches
        
        for val_loader in val_loader_list:
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
                outputs = self.backbone(inputs)
                loss = self.loss(outputs, labels)
            
                val_loss += (loss.item() / len(inputs))
            
                pred = outputs.argmax(dim=1)
                val_acc += (pred == labels).float().sum()
            
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.softmax(dim=1).detach().cpu().numpy())  # Use detach() before numpy()
        
        total_val_len = sum([len(val_loader.dataset) for val_loader in val_loader_list])
        val_acc /= total_val_len
        torch.cuda.empty_cache()
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Calculate AUROC
        try:
            auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except ValueError:
            auroc = None  # Handle case where AUROC cannot be computed
        
        self.backbone.train()
        
        return val_acc.item(), val_loss, auroc, conf_matrix

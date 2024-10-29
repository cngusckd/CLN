import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # Create a bar plot for buffer distribution
        fig, ax = plt.subplots()
        ax.bar(class_distribution.keys(), class_distribution.values())
        ax.set_title(f'Task {self.current_task_index} Buffer Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels
        
        # Log training metrics with wandb
        self.wandb.log({
            f'Task_{self.current_task_index}_TRAIN_buffer_distribution': self.wandb.Image(fig),  # Log bar plot as image
            f'Task_{self.current_task_index}_TRAIN_loss': loss,
        })
        
        plt.close(fig)  # Close the figure to free memory
        
        return None
    
    def wandb_eval_logger(self, val_acc, val_loss, auroc, conf_matrix, task_idx):
        
        # Calculate buffer distribution in buffer
        class_counts = torch.bincount(self.buffer.labels[:self.buffer.__len__()].squeeze(), minlength = self.cfg.nclasses)
        class_distribution = {f'class_{i}': class_counts[i].item() for i in range(self.cfg.nclasses)}

        # Create a bar plot for buffer distribution
        fig, ax = plt.subplots()
        ax.bar(class_distribution.keys(), class_distribution.values())
        ax.set_title(f'Task {self.current_task_index} Evaluation Buffer Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels
        
        # Create a heatmap for confusion matrix
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f'Task {self.current_task_index} Confusion Matrix')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        
        # Log evaluation metrics with wandb
        self.wandb.log({
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_buffer_distribution': self.wandb.Image(fig),  # Log bar plot as image
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_acc': val_acc,
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_loss': val_loss,
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_auroc': auroc,
            f'Task_{self.current_task_index}_about_{task_idx}_EVAL_conf_matrix': self.wandb.Image(fig_cm)  # Log confusion matrix as image
        })
        
        plt.close(fig)  # Close the figure to free memory
        plt.close(fig_cm)  # Close the confusion matrix figure to free memory
        
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

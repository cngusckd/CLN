import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from model.resnet import resnet18, resnet34, resnet50
from model.buffer import DefaultBuffer

class CL_MODEL(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = cfg.device
        
        if cfg.backbone == 'resnet18':
            self.backbone = resnet18(nclasses=cfg.nclasses, nf=cfg.image_shape[0]).to(self.device)
        elif cfg.backbone == 'resnet34':
            self.backbone = resnet34(nclasses=cfg.nclasses, nf=cfg.image_shape[0]).to(self.device)
        elif cfg.backbone == 'resnet50':
            self.backbone = resnet50(nclasses=cfg.nclasses, nf=cfg.image_shape[0]).to(self.device)
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")
        
        self.optimizer = self.get_optimizer()
        self.loss = self.get_loss_func()
        self.current_task_index = 0
        self.buffer = DefaultBuffer(cfg)
        self.wandb = wandb
    
    def __deepcopy__(self, memo):
        # Create a shallow copy of the object without deepcopying problematic modules
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy the attributes manually
        for k, v in self.__dict__.items():
            if k in ['wandb', 'optimizer', 'loss']:  # Exclude these attributes
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        
        return result
    
    def wandb_train_logger(self, loss):
        class_counts = torch.bincount(self.buffer.labels[:self.buffer.__len__()].squeeze(), minlength=self.cfg.nclasses)
        class_distribution = {f'class_{i}': class_counts[i].item() for i in range(self.cfg.nclasses)}
        
        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size
        ax.bar(class_distribution.keys(), class_distribution.values())
        ax.set_title(f'Task {self.current_task_index} Buffer Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        
        # Rotate x-axis labels and adjust font size
        plt.xticks(rotation=90, fontsize=8)
        
        # Optionally, display only every nth label to reduce clutter
        for label in ax.get_xticklabels()[::2]:  # Show every 2nd label
            label.set_visible(False)
        
        self.wandb.log({
            f'Task_{self.current_task_index}_TRAIN_buffer_distribution': self.wandb.Image(fig),
            f'Task_{self.current_task_index}_TRAIN_loss': loss,
        })
        
        plt.close(fig)
    
    def wandb_eval_logger(self, val_acc, val_loss, auroc, conf_matrix, all_labels, all_probs, task_idx):
        class_counts = torch.bincount(self.buffer.labels[:self.buffer.__len__()].squeeze(), minlength=self.cfg.nclasses)
        class_distribution = {f'class_{i}': class_counts[i].item() for i in range(self.cfg.nclasses)}

        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size
        ax.bar(class_distribution.keys(), class_distribution.values())
        ax.set_title(f'Task {self.current_task_index} Evaluation Buffer Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        
        # Rotate x-axis labels and adjust font size
        plt.xticks(rotation=90, fontsize=8)
        
        # Optionally, display only every nth label to reduce clutter
        for label in ax.get_xticklabels()[::2]:  # Show every 2nd label
            label.set_visible(False)
        
        fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    annot_kws={"size": 2},  # Adjust font size for annotations
                    cbar_kws={"shrink": 0.75})  # Adjust color bar size
        ax_cm.set_title(f'Task {self.current_task_index} Confusion Matrix')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        plt.xticks(rotation=45, fontsize=6)  # Adjust font size for x-axis labels
        plt.yticks(rotation=0, fontsize=6)  # Adjust font size for y-axis labels
        
        # Convert all_probs and all_labels to NumPy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Create ROC curve plot for each class
        fig_roc, ax_roc = plt.subplots()
        for i in range(self.current_task_index * (self.cfg.nclasses // self.cfg.num_increments)):
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, i], pos_label=i)
            # Ensure all_labels == i is a boolean array
            binary_labels = (all_labels == i).astype(int)
            
            # Check if there are at least two classes present
            if len(np.unique(binary_labels)) > 1:
                auroc_score = roc_auc_score(binary_labels, all_probs[:, i])
                ax_roc.plot(fpr, tpr, label=f'Class {i} AUROC = {auroc_score:.2f}')
            else:
                ax_roc.plot(fpr, tpr, label=f'Class {i} AUROC = N/A (single class)')

        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title(f'Task {self.current_task_index} ROC Curve')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc='lower right')
        
        self.wandb.log({
            f'Task_{self.current_task_index-1}_EVAL_buffer_distribution': self.wandb.Image(fig),
            f'Task_EVAL_acc': val_acc,
            f'Task_EVAL_loss': val_loss,
            f'Task_EVAL_auroc': auroc,
            f'Task_{self.current_task_index-1}_EVAL_conf_matrix': self.wandb.Image(fig_cm),
            f'Task_{self.current_task_index-1}_EVAL_roc_curve': self.wandb.Image(fig_roc)  # Log ROC curve as image
        })
        
        plt.close(fig)
        plt.close(fig_cm)
        plt.close(fig_roc)
    
    def get_parameters(self):
        return self.backbone.parameters()
    
    def get_optimizer(self):
        if self.cfg.optim == 'sgd':
            return torch.optim.SGD(params=self.get_parameters(),
                                   lr=self.cfg.lr,
                                   momentum=self.cfg.momentum)
        else:
            raise NotImplementedError
        
    def get_loss_func(self):
        return torch.nn.CrossEntropyLoss()
    
    def get_grads(self) -> torch.Tensor:
        grads = []
        for pp in list(self.backbone.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    
    def train_task(self):
        raise NotImplementedError
    
    def eval_task(self, val_loader_list):
        self.backbone.eval()
        val_loss = 0
        val_acc = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
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
                all_probs.extend(outputs.softmax(dim=1).detach().cpu().numpy())
        
        total_val_len = sum([len(val_loader.dataset) for val_loader in val_loader_list])
        val_acc /= total_val_len
        torch.cuda.empty_cache()
        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        try:
            auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except ValueError:
            auroc = None
        
        self.backbone.train()
        
        return val_acc.item(), val_loss, auroc, conf_matrix, all_labels, all_probs

"""
Pytorch Implementation of thr focal loss taken from
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
Credits : https://github.com/clcarwin

Only modified to add the option to ignore a label
"""
"""
FocalLoss compatible avec PaPsLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_label=None):
        super(FocalLoss, self).__init__()
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
            
        self.gamma = gamma
        self.ignore_label = ignore_label if ignore_label is not None else -100
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - logits
            targets: (N,) - ground truth classes
        """
            # Ajouter ces lignes de debug pour comprendre le problème
        #print(f"DEBUG - Inputs shape: {inputs.shape}")
        #print(f"DEBUG - Targets shape: {targets.shape}")
        #print(f"DEBUG - Targets dtype: {targets.dtype}")
        #print(f"DEBUG - Targets unique values: {torch.unique(targets)}")
    
    # Corriger le format des targets si nécessaire
        if targets.dim() > 1:
            targets = targets.squeeze()  # Supprimer les dimensions unitaires
        
    # Si targets est encore multi-dimensionnel, le flattener
        if targets.dim() > 1:
            targets = targets.view(-1)
            inputs = inputs.view(inputs.size(0), -1)  # Adapter inputs aussi
    
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            ignore_index=self.ignore_label
        )
        
        # Calculer les probabilités pt
        pt = torch.exp(-ce_loss)
        
        # Terme de focusing (1-pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Appliquer les poids alpha si fournis
        if self.alpha is not None:
            # Masque pour targets valides
            valid_mask = (targets != self.ignore_label)
            
            if valid_mask.any():
                # Extraire alpha pour chaque target selon sa classe
                alpha_t = self.alpha[targets[valid_mask]]
                
                # Appliquer alpha seulement aux éléments valides
                focal_loss_weighted = torch.zeros_like(focal_loss)
                focal_loss_weighted[valid_mask] = alpha_t * focal_loss[valid_mask]
                focal_loss_weighted[~valid_mask] = focal_loss[~valid_mask]
                focal_loss = focal_loss_weighted
        
        # Réduction finale
        valid_elements = (targets != self.ignore_label).sum()
        if valid_elements > 0:
            return focal_loss.sum() / valid_elements
        else:
            return focal_loss.sum()
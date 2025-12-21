import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with Homoscedastic Uncertainty Weighting.
    Loss = sum( 1/(2*sigma^2) * L_i + log(sigma) )
    We learn log_var = log(sigma^2) for numerical stability.
    """
    def __init__(self, num_tasks=4):
        super(MultiTaskLoss, self).__init__()
        # log_vars are trainable parameters: [part, side, price, qty]
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # Losses
        self.loss_part = FocalLoss(alpha=0.75, gamma=2.0) # Handle imbalance
        self.loss_side = nn.BCEWithLogitsLoss()
        self.loss_reg = nn.MSELoss()

    def forward(self, preds, targets):
        """
        preds: (l_part, l_side, l_price, l_qty)
        targets: (y_part, y_side, y_price, y_qty)
        """
        l_part_pred, l_side_pred, l_price_pred, l_qty_pred = preds
        y_part, y_side, y_price, y_qty = targets
        
        # --- Task 1: Participation (Always computed) ---
        L1 = self.loss_part(l_part_pred.squeeze(), y_part)
        
        # Mask for conditional tasks (only when prop_participated=1)
        mask = y_part == 1
        
        if mask.sum() == 0:
            # If no participation in batch, only return L1 (weighted)
            # We still need to populate other losses for logging, but they are 0
            loss = self._weighted_loss(L1, 0)
            return loss, L1.item(), 0.0, 0.0, 0.0
        
        # --- Task 2: Side (Buy/Sell) ---
        L2 = self.loss_side(l_side_pred.squeeze()[mask], y_side[mask])
        
        # --- Task 3: Price Delta (Regression) ---
        L3 = self.loss_reg(l_price_pred.squeeze()[mask], y_price[mask])
        
        # --- Task 4: Quantity (Regression) ---
        L4 = self.loss_reg(l_qty_pred.squeeze()[mask], y_qty[mask])
        
        # Combine Loss using Uncertainty Weighting
        precision1 = torch.exp(-self.log_vars[0])
        loss = precision1 * L1 + 0.5 * self.log_vars[0]
        
        precision2 = torch.exp(-self.log_vars[1])
        loss += precision2 * L2 + 0.5 * self.log_vars[1]
        
        precision3 = torch.exp(-self.log_vars[2])
        loss += precision3 * L3 + 0.5 * self.log_vars[2]
        
        precision4 = torch.exp(-self.log_vars[3])
        loss += precision4 * L4 + 0.5 * self.log_vars[3]
        
        return loss, L1.item(), L2.item(), L3.item(), L4.item()

    def _weighted_loss(self, L1, idx):
        # Helper for single task backprop if needed, though usually we sum all
        precision = torch.exp(-self.log_vars[idx])
        return precision * L1 + 0.5 * self.log_vars[idx]
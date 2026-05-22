import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, init_temperature=0.07, learnable_temperature=True):
        """
        InfoNCE (Information Noise-Contrastive Estimation) Loss.
        Aligns the predicted embedding with the target embedding using in-batch negatives.
        """
        super().__init__()
        if learnable_temperature:
            # Initialize as a learnable parameter (standard practice in modern VLMs like CLIP)
            self.temperature = nn.Parameter(torch.tensor([init_temperature]))
        else:
            self.register_buffer("temperature", torch.tensor([init_temperature]))

    def forward(self, predicted_embeds, target_embeds):
        """
        Args:
            predicted_embeds: S_Y_hat from the Predictor, shape [Batch, Dim]
            target_embeds: S_Y from the Y-Encoder, shape [Batch, Dim]
        """
        batch_size = predicted_embeds.shape[0]
        
        # 1. L2 Normalize the embeddings
        # This projects the vectors onto a unit hypersphere, making dot products equivalent to cosine similarity
        pred_norm = F.normalize(predicted_embeds, p=2, dim=1)
        target_norm = F.normalize(target_embeds, p=2, dim=1)
        
        # Clamp temperature to prevent division by zero or numerical instability
        temp = torch.clamp(self.temperature, min=1e-3, max=100.0)
        
        # 2. Compute similarity matrix (Logits)
        # Shape: [Batch, Batch]
        logits = torch.matmul(pred_norm, target_norm.T) / temp
        
        # 3. Create Targets
        # The positive pairs are on the diagonal (0-to-0, 1-to-1, ..., B-to-B)
        labels = torch.arange(batch_size, dtype=torch.long, device=predicted_embeds.device)
        
        # 4. Compute Symmetric Cross Entropy
        # Predictor-to-Target Loss (Alignment of Prediction to Target)
        loss_p2t = F.cross_entropy(logits, labels)
        
        # Target-to-Predictor Loss (Alignment of Target to Prediction)
        loss_t2p = F.cross_entropy(logits.T, labels)
        
        # Symmetric average
        return (loss_p2t + loss_t2p) / 2.0


class VLJepaLoss(nn.Module):
    def __init__(self, init_temperature=0.07, l2_reg_weight=1e-4):
        """
        Combines InfoNCE (Alignment) with an L2 penalty (Regularization) 
        as specified in the VL-JEPA architecture diagram.
        """
        super().__init__()
        self.alignment_loss = InfoNCELoss(init_temperature=init_temperature)
        self.l2_reg_weight = l2_reg_weight

    def forward(self, predicted_embeds, target_embeds):
        # 1. Alignment Loss (InfoNCE)
        info_nce = self.alignment_loss(predicted_embeds, target_embeds)
        
        # 2. Regularization Loss
        # Penalizes excessively large embedding magnitudes to stabilize the shared embedding space
        reg_pred = torch.mean(predicted_embeds ** 2)
        reg_target = torch.mean(target_embeds ** 2)
        regularization = self.l2_reg_weight * (reg_pred + reg_target)
        
        # Total Loss
        total_loss = info_nce + regularization
        
        return total_loss, {"loss": total_loss, "info_nce": info_nce, "regularization": regularization}
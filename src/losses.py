import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Computes the InfoNCE loss.
        
        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim], 
                                     where features[:batch_size] are the representations of 
                                     the first set of augmented images, and features[batch_size:] 
                                     are the representations of the second set.
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Normalize features to have unit norm
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Get batch size
        batch_size = features.shape[0] // 2
        
        # Construct labels where each sample's positive pair is in the other view
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class VLJepaLoss(nn.Module):
    def __init__(self, init_temperature=0.07, l2_reg_weight=0.0, temperature=None):
        super().__init__()

        if temperature is None:
            temperature = init_temperature

        self.infonce = InfoNCE(temperature=temperature)
        self.l2_reg_weight = l2_reg_weight

    def forward(self, pred_emb, target_emb):
        """
        pred_emb:   [B, D] from Predictor
        target_emb: [B, D] from Y-Encoder
        """

        if pred_emb.dim() != 2 or target_emb.dim() != 2:
            raise ValueError(
                f"Expected pred_emb and target_emb to be 2D tensors [B, D], "
                f"got {pred_emb.shape} and {target_emb.shape}"
            )

        if pred_emb.shape != target_emb.shape:
            raise ValueError(
                f"pred_emb and target_emb must have the same shape, "
                f"got {pred_emb.shape} and {target_emb.shape}"
            )

        batch_size = pred_emb.shape[0]

        if batch_size < 1:
            raise ValueError("Expected at least one sample in the batch.")

        zero_loss = pred_emb.new_zeros(())

        if batch_size >= 2:
            # [2B, D]
            features = torch.cat([pred_emb, target_emb], dim=0)

            # SimCLR-style bidirectional InfoNCE
            info_nce = self.infonce(features)
            align_loss = zero_loss
            main_loss = info_nce
        else:
            info_nce = zero_loss
            align_loss = 1 - F.cosine_similarity(
                pred_emb,
                target_emb,
                dim=1,
            ).mean()
            main_loss = align_loss

        if self.l2_reg_weight > 0:
            reg_pred = torch.mean(pred_emb ** 2)
            reg_target = torch.mean(target_emb ** 2)
            regularization = self.l2_reg_weight * (reg_pred + reg_target)
        else:
            regularization = zero_loss

        total_loss = main_loss + regularization

        return total_loss, {
            "loss": total_loss.detach(),
            "info_nce": info_nce.detach(),
            "align_loss": align_loss.detach(),
            "regularization": regularization.detach(),
        }

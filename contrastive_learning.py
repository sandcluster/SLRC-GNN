import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    监督对比学习损失
    参考论文: Supervised Contrastive Learning
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels=None, mask=None):
        """
        计算监督对比损失
        
        Args:
            features: 特征向量, 形状为 [batch_size, feature_dim]
            labels: 标签, 形状为 [batch_size]
            mask: 自定义的掩码, 形状为 [batch_size, batch_size], 可选
            
        Returns:
            对比损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 特征归一化
        features = F.normalize(features, dim=1)
        
        # 计算特征点积相似度矩阵 (cosine similarity)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标识矩阵(对角线为1)
        eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # 对角线设为极小值(排除自身)
        sim_matrix = sim_matrix.masked_fill(eye_mask, -float('inf'))
        
        # 若没有提供标签，则使用无监督的方式
        if labels is None:
            # 简单的InfoNCE损失
            logits = sim_matrix
            positive_mask = eye_mask  # 对角线为正例(自身)
            labels = torch.arange(batch_size, device=device)
            
        else:
            # 创建正例掩码: 相同标签的样本被视为正例
            labels = labels.contiguous().view(-1, 1)
            positive_mask = torch.eq(labels, labels.T).to(device)
            
            # 排除自身
            positive_mask = positive_mask & (~eye_mask)
            
            # 如果一个样本没有正例，我们用自身作为正例
            pos_per_sample = positive_mask.sum(1)
            zero_pos_idx = (pos_per_sample == 0)
            if torch.any(zero_pos_idx):
                positive_mask[zero_pos_idx, zero_pos_idx] = True
            
            # 计算对比损失
            logits = sim_matrix
        
        # 计算log_prob
        exp_logits = torch.exp(logits)
        
        # 计算正例的log-likelihood
        pos_exp_sum = torch.sum(exp_logits * positive_mask, dim=1, keepdim=True)
        log_prob = torch.log(pos_exp_sum) - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # 计算每个样本的平均损失
        mean_log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        
        # 返回批次平均损失
        loss = -mean_log_prob.mean()
        
        return loss

class SessionContrastiveLoss(nn.Module):
    """
    会话数据的对比学习损失
    融合序列表征和物品表征的对比学习
    """
    def __init__(self, temperature=0.07, lambda_cl=0.1):
        super(SessionContrastiveLoss, self).__init__()
        self.temp = temperature
        self.lambda_cl = lambda_cl
        self.sup_con_loss = SupConLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, seq_repr, augmented_repr, scores, targets):
        """
        计算会话推荐的对比学习损失
        
        Args:
            seq_repr: 原始会话表征, [batch_size, hidden_dim]
            augmented_repr: 增强后的会话表征, [batch_size, hidden_dim]
            scores: 预测分数, [batch_size, num_items]
            targets: 目标物品, [batch_size]
            
        Returns:
            总损失 = CE损失 + lambda * 对比损失
        """
        # 标准交叉熵损失
        ce_loss = self.ce_loss(scores, targets)
        
        # 将原始表征和增强表征拼接
        combined_features = torch.cat([seq_repr.unsqueeze(1), augmented_repr.unsqueeze(1)], dim=1)
        # 重塑为 [batch_size*2, hidden_dim]
        batch_size = seq_repr.shape[0]
        features = combined_features.view(batch_size*2, -1)
        
        # 创建对应的标签(每对表征具有相同的标签)
        labels = torch.arange(batch_size, device=seq_repr.device).repeat_interleave(2)
        
        # 计算对比损失
        cl_loss = self.sup_con_loss(features, labels)
        
        # 总损失
        total_loss = ce_loss + self.lambda_cl * cl_loss
        
        return total_loss, ce_loss, cl_loss
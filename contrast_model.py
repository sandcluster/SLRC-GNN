import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model import CombineGraph, trans_to_cuda


class ContrastGCEGNN(CombineGraph):
    def __init__(self, opt, num_node, adj_all, num):
        super(ContrastGCEGNN, self).__init__(opt, num_node, adj_all, num)
        
        # 对比学习相关超参数
        self.temperature_base = opt.temperature  # 基础温度参数
        self.temperature_min = 0.05  # 最小温度值
        self.temperature_decay = 0.005  # 温度衰减率
        self.adapt_temp = opt.adapt_temp if hasattr(opt, 'adapt_temp') else True  # 是否使用自适应温度
        
        self.contrast_weight = opt.contrast_weight  # 对比损失的权重
        self.cl_dropout = opt.cl_dropout  # 用于数据增强的dropout率
        self.semantic_weight = opt.semantic_weight if hasattr(opt, 'semantic_weight') else 0.7  # 语义权重
        
        # 投影头，用于将特征映射到对比学习空间
        self.proj_local = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        
        self.proj_global = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        
        # 创建自适应温度参数
        self.register_buffer('temperature', torch.tensor(self.temperature_base))
        self.register_buffer('step_count', torch.tensor(0))

    def update_temperature(self):
        """更新自适应温度参数"""
        if self.adapt_temp:
            # 步数增加
            self.step_count += 1
            # 随训练进行逐渐降低温度
            decay = self.temperature_decay * self.step_count
            # 使用torch.tensor而不是直接赋值浮点数
            new_temp = max(self.temperature_base * math.exp(-decay), self.temperature_min)
            self.temperature.copy_(torch.tensor(new_temp, device=self.temperature.device))
        
    def random_mask(self, hidden, mask, mask_prob=None):
        """
        随机掩码部分项目，但保留最后一个项目
        
        Args:
            hidden: 输入的会话表示 [batch_size, seq_len, hidden_size]
            mask: 会话掩码 [batch_size, seq_len]
            mask_prob: 掩码概率，如果为None则使用self.cl_dropout
            
        Returns:
            augmented_hidden: 增强后的会话表示
        """
        if mask_prob is None:
            mask_prob = self.cl_dropout
            
        batch_size, seq_len, hidden_size = hidden.shape
        
        # 创建有效项掩码
        valid_mask = mask.float().unsqueeze(-1) > 0
        
        # 为每个会话找到最后一个有效项目的位置
        last_item_pos = torch.zeros(batch_size, seq_len, device=hidden.device)
        for i in range(batch_size):
            valid_items = torch.where(mask[i] > 0)[0]
            if len(valid_items) > 0:
                last_pos = valid_items[-1].item()
                last_item_pos[i, last_pos] = 1.0
        
        last_item_mask = last_item_pos.unsqueeze(-1) > 0
        
        # 创建随机掩码，但排除最后一个项目
        random_mask = torch.rand(batch_size, seq_len, 1, device=hidden.device) < mask_prob
        
        # 只在原始掩码为1且不是最后一项的地方应用随机掩码
        effective_mask = random_mask & valid_mask & (~last_item_mask)
        
        # 应用掩码
        zeros = torch.zeros_like(hidden)
        augmented_hidden = torch.where(effective_mask, zeros, hidden)
        
        return augmented_hidden
    
    def noise_injection(self, hidden, mask, noise_level=0.1):
        """
        向嵌入添加高斯噪声
        
        Args:
            hidden: 输入的会话表示 [batch_size, seq_len, hidden_size]
            mask: 会话掩码 [batch_size, seq_len]
            noise_level: 噪声强度
            
        Returns:
            augmented_hidden: 增强后的会话表示
        """
        # 生成噪声
        noise = torch.randn_like(hidden) * noise_level
        
        # 只对有效位置添加噪声
        valid_positions = mask.unsqueeze(-1).expand_as(hidden).float()
        augmented_hidden = hidden + noise * valid_positions
        
        return augmented_hidden
    
    def subsequence_sampling(self, hidden, mask):
        """
        随机提取连续子序列
        
        Args:
            hidden: 输入的会话表示 [batch_size, seq_len, hidden_size]
            mask: 会话掩码 [batch_size, seq_len]
            
        Returns:
            augmented_hidden: 增强后的会话表示
            new_mask: 更新后的掩码
        """
        batch_size, seq_len, hidden_size = hidden.shape
        augmented_hidden = torch.zeros_like(hidden)
        new_mask = torch.zeros_like(mask)
        
        for i in range(batch_size):
            valid_len = torch.sum(mask[i]).item()
            if valid_len > 2:
                # 随机选择子序列的起始位置和长度
                max_start = valid_len // 2  # 确保子序列不会太短
                start = torch.randint(0, max_start, (1,)).item()
                length = torch.randint(valid_len // 2, valid_len - start, (1,)).item()
                
                # 复制子序列到增强后的序列
                augmented_hidden[i, :length] = hidden[i, start:start+length]
                # 更新mask
                new_mask[i, :length] = 1
            else:
                # 序列太短，不进行裁剪
                augmented_hidden[i] = hidden[i]
                new_mask[i] = mask[i]
        
        return augmented_hidden, new_mask
    
    def temporal_perturbation(self, hidden, mask):
        """
        局部顺序扰动，保持大体顺序但对局部进行重排
        
        Args:
            hidden: 输入的会话表示 [batch_size, seq_len, hidden_size]
            mask: 会话掩码 [batch_size, seq_len]
            
        Returns:
            augmented_hidden: 增强后的会话表示
        """
        batch_size, seq_len, hidden_size = hidden.shape
        augmented_hidden = hidden.clone()
        
        for i in range(batch_size):
            valid_len = torch.sum(mask[i]).item()
            if valid_len > 3:
                # 保留最后一个项目
                last_item_pos = torch.where(mask[i] > 0)[0][-1].item()
                
                # 将序列分成几个块，但排除最后一个项目
                valid_len_for_perm = last_item_pos  # 不包括最后一个项目
                if valid_len_for_perm > 3:
                    num_blocks = min(valid_len_for_perm // 2, 3)
                    block_size = valid_len_for_perm // num_blocks
                    
                    for b in range(num_blocks):
                        start = b * block_size
                        end = (b+1) * block_size if b < num_blocks-1 else valid_len_for_perm
                        
                        # 在块内随机打乱
                        if end - start > 1:
                            perm = torch.randperm(end - start)
                            augmented_hidden[i, start:end] = hidden[i, start:end][perm]
        
        return augmented_hidden
    
    def data_augmentation(self, hidden, mask, augmentation_type='view1'):
        """
        对输入的会话序列表示进行数据增强
        
        Args:
            hidden: 输入的会话表示
            mask: 会话掩码
            augmentation_type: 增强类型
            
        Returns:
            augmented_hidden: 增强后的会话表示
            new_mask: 可能更新的掩码 (对于subsequence)
        """
        new_mask = mask.clone()
        
        if augmentation_type == 'dropout':
            # 随机丢弃部分特征
            dropout = nn.Dropout(self.cl_dropout)
            augmented_hidden = dropout(hidden)
            return augmented_hidden, new_mask
        
        elif augmentation_type == 'mask':
            # 随机掩码部分项目，但保留最后一个项目
            augmented_hidden = self.random_mask(hidden, mask)
            return augmented_hidden, new_mask
        
        elif augmentation_type == 'noise':
            # 添加噪声
            augmented_hidden = self.noise_injection(hidden, mask)
            return augmented_hidden, new_mask
        
        elif augmentation_type == 'subsequence':
            # 随机提取子序列
            return self.subsequence_sampling(hidden, mask)
        
        elif augmentation_type == 'reorder':
            # 局部顺序扰动
            augmented_hidden = self.temporal_perturbation(hidden, mask)
            return augmented_hidden, new_mask
        
        elif augmentation_type == 'view1':
            # 第一视角：结合随机掩码和噪声注入
            if torch.rand(1).item() < 0.5:
                augmented_hidden = self.random_mask(hidden, mask)
            else:
                augmented_hidden = self.noise_injection(hidden, mask)
            return augmented_hidden, new_mask
        
        elif augmentation_type == 'view2':
            # 第二视角：结合子序列采样和局部重排
            if torch.rand(1).item() < 0.5:
                return self.subsequence_sampling(hidden, mask)
            else:
                augmented_hidden = self.temporal_perturbation(hidden, mask)
                return augmented_hidden, new_mask
                
        else:
            # 默认不做增强
            return hidden.clone(), new_mask
        
    def contrastive_loss(self, local_repr, global_repr, local_mask, global_mask):
        """
        计算局部和全局表示之间的对比损失，加入语义权重
        
        Args:
            local_repr: 局部表示 [batch_size, seq_len, dim]
            global_repr: 全局表示 [batch_size, seq_len, dim]
            local_mask: 局部会话掩码 [batch_size, seq_len]
            global_mask: 全局会话掩码 [batch_size, seq_len]
            
        Returns:
            loss: 对比损失
        """
        batch_size, seq_len, dim = local_repr.shape
        
        # 使用投影头投影特征
        z_local = self.proj_local(local_repr)
        z_global = self.proj_global(global_repr)
        
        # 对特征进行L2归一化
        z_local = F.normalize(z_local, dim=-1)
        z_global = F.normalize(z_global, dim=-1)
        
        # 将序列展平为样本集合
        z_local_flat = z_local.view(-1, dim)  # [batch_size*seq_len, dim]
        z_global_flat = z_global.view(-1, dim)  # [batch_size*seq_len, dim]
        local_mask_flat = local_mask.view(-1).bool()  # [batch_size*seq_len]
        global_mask_flat = global_mask.view(-1).bool()  # [batch_size*seq_len]
        
        # 合并有效掩码，只考虑两个视图中都有效的位置
        valid_mask = local_mask_flat & global_mask_flat
        
        # 只选择有效的(非padding)位置
        z_local_valid = z_local_flat[valid_mask]  # [n_valid, dim]
        z_global_valid = z_global_flat[valid_mask]  # [n_valid, dim]
        
        n_valid = z_local_valid.shape[0]
        if n_valid <= 1:  # 没有足够的样本进行对比学习
            return torch.tensor(0.0, device=local_repr.device)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_local_valid, z_global_valid.t()) / self.temperature  # [n_valid, n_valid]
        
        # 创建标签，对角线位置是正样本对
        labels = torch.arange(n_valid, device=sim_matrix.device)
        
        # 创建语义权重矩阵，减轻不同项目间的对齐要求
        if self.semantic_weight < 1.0:
            # 计算项目特征相似度
            item_sim = torch.matmul(z_local_valid, z_local_valid.t())  # [n_valid, n_valid]
            # 归一化到[0,1]
            item_sim = (item_sim + 1) / 2
            # 创建权重矩阵，相似度高的项目对权重大
            weights = torch.ones_like(item_sim) * self.semantic_weight
            # 对角线权重为1，即正样本对权重最大
            weights.fill_diagonal_(1.0)
            # 根据项目相似度动态调整权重
            weights = weights + (1 - weights) * item_sim
            
            # 修改InfoNCE损失以考虑权重
            exp_sim = torch.exp(sim_matrix)
            # 对角线位置的正样本概率，分子
            pos_probs = torch.diagonal(exp_sim)
            # 所有样本对的概率，分母
            neg_probs = (exp_sim * weights).sum(dim=1)
            # 计算损失 -log(pos_prob / weighted_sum)
            loss = -torch.log(pos_probs / neg_probs).mean()
        else:
            # 标准InfoNCE损失
            loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)
        
        # 原始局部和全局表示
        h_local_orig = self.local_agg(h, adj, mask_item)
        
        # 全局表示
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len
        
        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))
            
        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors
        
        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        
        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
            
        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                   neighbor_vector=entity_vectors[hop+1].view(shape),
                                   masks=None,
                                   batch_size=batch_size,
                                   neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                   extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
            
        h_global_orig = entity_vectors[0].view(batch_size, seqs_len, self.dim)
        
        # 存储原始表示，用于对比学习
        self.h_local_orig = h_local_orig
        self.h_global_orig = h_global_orig
        
        # 在训练模式下生成对比损失
        if self.training:
            # 更新温度参数
            self.update_temperature()
            
            # 第一视角 - 保留结构的增强: 随机掩码或噪声注入
            h_view1, mask_view1 = self.data_augmentation(h_local_orig, mask_item, 'view1')
            
            # 第二视角 - 改变结构的增强: 子序列采样或局部重排
            h_view2, mask_view2 = self.data_augmentation(h_global_orig, mask_item, 'view2')
            
            # 计算对比损失
            contrast_loss = self.contrastive_loss(h_view1, h_view2, mask_view1, mask_view2)
        else:
            contrast_loss = torch.tensor(0.0, device=h_local_orig.device)
        
        # 用于预测的组合表示
        h_local = F.dropout(h_local_orig, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global_orig, self.dropout_global, training=self.training)
        output = h_local + h_global
        
        return output, contrast_loss

    # 下面是一个训练步骤的示例函数，供参考
    def train_step(self, optimizer, inputs, adj, mask_item, item, targets):
        """
        单次训练步骤
        
        Args:
            optimizer: 优化器
            inputs: 输入序列
            adj: 邻接矩阵
            mask_item: 项目掩码
            item: 目标项目
            targets: 真实标签
            
        Returns:
            loss: 总损失
            rec_loss: 推荐损失
            cl_loss: 对比损失
        """
        optimizer.zero_grad()
        
        # 前向传播，获取预测和对比损失
        scores, cl_loss = self.forward(inputs, adj, mask_item, item)
        
        # 计算推荐任务的损失
        rec_loss = self.loss_function(scores, targets)
        
        # 总损失 = 推荐损失 + 对比损失 * 权重
        loss = rec_loss + self.contrast_weight * cl_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        
        # 更新参数
        optimizer.step()
        
        return loss.item(), rec_loss.item(), cl_loss.item()
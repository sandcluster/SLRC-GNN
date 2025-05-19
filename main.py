import time
import argparse
import pickle
import os
import numpy as np
import torch
from model import *
from utils import *

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica_small', help='小样本数据集路径')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=8)
parser.add_argument('--n_sample', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--lr_dc', type=float, default=0.1, help='学习率衰减')
parser.add_argument('--lr_dc_step', type=int, default=3, help='学习率衰减步数')
parser.add_argument('--l2', type=float, default=1e-5, help='L2正则化')
parser.add_argument('--n_iter', type=int, default=3)  # 对于diginetica，使用2
parser.add_argument('--dropout_gcn', type=float, default=0.2)  # 对于diginetica，使用0.2
parser.add_argument('--dropout_local', type=float, default=0.0)  # 对于diginetica，使用0.0
parser.add_argument('--dropout_global', type=float, default=0.5)
parser.add_argument('--validation', action='store_true', help='是否使用验证集')
parser.add_argument('--valid_portion', type=float, default=0.1, help='验证集比例')
parser.add_argument('--alpha', type=float, default=0.2, help='LeakyReLU的alpha值')
parser.add_argument('--patience', type=int, default=2)

parser.add_argument('--gcnii_alpha', type=float, default=0.5, help='GCNII初始残差连接强度')
parser.add_argument('--gcnii_beta', type=float, default=0.1, help='GCNII身份映射强度')
parser.add_argument('--use_residual', action='store_true', default=True, help='是否使用GCNII残差连接')

opt = parser.parse_args()

def main():
    init_seed(2025)  # 使用固定种子以便结果可重现

    # 加载数据集并确定节点数量
    train_data = pickle.load(open(os.path.join(opt.dataset, 'train.txt'), 'rb'))
    test_data = pickle.load(open(os.path.join(opt.dataset, 'test.txt'), 'rb'))
    
    # 确定商品数量
    all_items = set()
    for seq, _ in zip(train_data[0], train_data[1]):
        all_items.update(seq)
    all_items.update([train_data[1][i] for i in range(len(train_data[1]))])
    num_node = max(all_items) + 1  # +1 因为商品ID从1开始
    
    print(f"数据集中商品总数: {num_node}")
    
    # 加载邻接表和权重
    adj = pickle.load(open(os.path.join(opt.dataset, f'adj_{opt.n_sample_all}.pkl'), 'rb'))
    num = pickle.load(open(os.path.join(opt.dataset, f'num_{opt.n_sample_all}.pkl'), 'rb'))
    
    # 如果实际节点数与加载的邻接表大小不一致，可能需要调整
    if len(adj) < num_node:
        print(f"警告: 邻接表大小({len(adj)})小于节点数({num_node})，调整大小...")
        adj.extend([[] for _ in range(num_node - len(adj))])
        num.extend([[] for _ in range(num_node - len(num))])
    
    # 准备数据集
    train_data = Data(train_data)
    test_data = Data(test_data)

    # 处理邻接表
    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    
    # 创建模型
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    print(opt)
    start = time.time()
    best_result = [0, 0, 0, 0]  # 扩展为4个指标 [recall@10, mrr@10, recall@20, mrr@20]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics = train_test(model, train_data, test_data)  # 获取4个指标
        recall10, mrr10, recall20, mrr20 = metrics
        flag = 0
        for i, metric in enumerate(metrics):
            if metric >= best_result[i]:
                best_result[i] = metric
                best_epoch[i] = epoch
                flag = 1
        print('当前结果:')
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f' % (recall10, mrr10))
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f' % (recall20, mrr20))
        print('最佳结果:')
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[2], best_result[3], best_epoch[2], best_epoch[3]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break

    print('-------------------------------------------------------')
    end = time.time()
    print("运行时间: %f 秒" % (end - start))


if __name__ == '__main__':
    main()
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica_small', help='处理后的小样本数据集')
parser.add_argument('--sample_num', type=int, default=8, help='每个节点采样的邻居数量')
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

# 加载训练序列
seq = pickle.load(open(os.path.join(dataset, 'all_train_seq.txt'), 'rb'))

# 确定商品总数
max_item_id = 0
for s in seq:
    max_item_id = max(max_item_id, max(s))
num = max_item_id + 1  # +1 因为商品ID从1开始

print(f"小样本数据集中商品总数: {num}")

# 初始化关系列表和邻接表
relation = []
adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

# 构建商品之间的关系
print("构建商品关系...")
for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):  # 考虑1, 2, 3步距离的关系
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

# 统计关系频次
print("统计关系频次...")
for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

# 初始化权重列表
weight = [[] for _ in range(num)]

# 根据频次排序并选择前N个邻居
print(f"为每个商品选择前{sample_num}个邻居...")
for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

# 截取指定数量的邻居
for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

# 保存邻接表和权重
print("保存结果...")
pickle.dump(adj, open(os.path.join(dataset, f'adj_{sample_num}.pkl'), 'wb'))
pickle.dump(weight, open(os.path.join(dataset, f'num_{sample_num}.pkl'), 'wb'))
print("图结构构建完成！")
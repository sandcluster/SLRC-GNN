import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica')
opt = parser.parse_args()   
print(opt)

dataset = 'C:/Users/charon/Desktop/final_codeworks/data/diginetica/train-item-views.csv'  # 假设数据文件名为diginetica.csv

print("-- Starting @ %s" % datetime.datetime.now())
with open(dataset, "r") as f:
    reader = csv.DictReader(f, delimiter=';')  # 根据实际分隔符调整
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = int(data['sessionId'])  # 根据实际列名调整
        if curdate and not curid == sessid:
            date = curdate
            sess_date[curid] = date
        curid = sessid

        item = int(data['itemId'])  # 根据实际列名调整
        curdate = float(data['timeframe'])  # 根据实际列名调整

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    
    # 最后一个会话
    date = float(data['timeframe'])
    sess_date[curid] = date
    
print('总点击记录数:', ctr)
print("-- 数据读取完成 @ %s" % datetime.datetime.now())

# 筛选出长度大于1的会话
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# 统计每个商品出现的次数
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

# 过滤出现次数少于5次的商品和长度小于2或大于30的会话
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2 or len(filseq) > 30:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# 排序会话按日期
dates = list(sess_date.items())
dates.sort(key=operator.itemgetter(1))

# 只取最新的10%数据
start_idx = int(len(dates) * 0.96)
recent_dates = dates[start_idx:]

# 按80%训练集和20%测试集拆分
split_idx = int(len(recent_dates) * 0.8)
tra_sess = recent_dates[:split_idx]
tes_sess = recent_dates[split_idx:]

print('训练集会话数:', len(tra_sess))
print('测试集会话数:', len(tes_sess))
print('训练集前3个会话:', tra_sess[:3])
print('测试集前3个会话:', tes_sess[:3])
print("-- 训练集和测试集拆分完成 @ %s" % datetime.datetime.now())

# 将训练会话转换为序列并重新编号商品，从1开始
item_dict = {}

def obtain_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # 不应该发生
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print('商品总数:', item_ctr-1)
    return train_ids, train_dates, train_seqs

# 转换测试会话为序列，忽略在训练集中未出现的商品
def obtain_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs

tra_ids, tra_dates, tra_seqs = obtain_tra()
tes_ids, tes_dates, tes_seqs = obtain_tes()

# 处理序列生成训练样本
def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

print('训练样本数:', len(tr_seqs))
print('测试样本数:', len(te_seqs))
print('训练样本示例:', tr_seqs[:3], tr_labs[:3])
print('测试样本示例:', te_seqs[:3], te_labs[:3])

# 计算平均序列长度
total_len = 0
for seq in tra_seqs:
    total_len += len(seq)
for seq in tes_seqs:
    total_len += len(seq)
print('平均序列长度:', total_len / (len(tra_seqs) + len(tes_seqs)))
print('总点击次数:', total_len)

# 创建输出目录
output_dir = 'diginetica_small'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存处理后的数据
pickle.dump(tra, open(os.path.join(output_dir, 'train.txt'), 'wb'))
pickle.dump(tes, open(os.path.join(output_dir, 'test.txt'), 'wb'))
pickle.dump(tra_seqs, open(os.path.join(output_dir, 'all_train_seq.txt'), 'wb'))

# 为了构建图，我们还需要创建一个脚本来构建小样本的图结构
print("-- 数据处理完成 @ %s" % datetime.datetime.now())
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def get_gaucs(data_feats, data_labels, valid_fold, model):
    n = len(valid_fold)
    dev_feats = data_feats[valid_fold]
    dev_labels = data_labels[valid_fold]
    user_indexes = dev_feats[:, 0]
    outputs = [[] for _ in range(7)]
    for i in range(n // 4096 + 1):
        output = model(dev_feats[i * 4096: (i + 1) * 4096])
        for i in range(7):
            outputs[i].append(output[i])
    gaucs = list()
    for i in range(7):
        preds = np.concatenate(outputs[i])
        labels = dev_labels[:, i]
        gauc = cal_group_auc(labels, preds, user_indexes)
        gaucs.append(gauc)
    return gaucs


def cal_group_auc(labels, preds, user_id_list):
    print('*' * 50)
    print("runing standard gauc computation...")
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag
    impression_total = 0
    total_auc = 0

    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc

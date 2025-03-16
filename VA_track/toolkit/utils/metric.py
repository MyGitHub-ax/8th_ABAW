import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score
from torch import nn

from ..globals import *

#自定义你的metric，按照下面的模板,注意labels和preds是二维数组，[[1],[2],[3],[4]],这样方便处理维度的label[[1，2，3，4],[2，3，4，5],[3，4，5，6],[4，5，6，7]
##########################################################
# def template(preds,labels):
#     return metric(labels,preds)


##########################################################

# 只返回 metric 值，用于模型筛选 
def gain_metric_from_results(eval_results, metric_list=[]):
    results={}
    for metric_name in metric_list:
        # 通过 globals() 动态查找并调用对应的函数
        metric_func = globals().get(f'{metric_name}metric')
        if metric_func:
            results[metric_name]=metric_func(eval_results['preds'],eval_results['labels'])
        else:
            raise ValueError(f"Metric function {metric_name}metric not found")

    return results

def MSEmetric(preds,labels):
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    return mean_squared_error(labels,preds)


def cccmetric(preds, labels):
    """
    输入要求:
    - preds: 形状为 (N, T, 2) 的数组，N=6693 个样本，T=1000 时间步，2=valence和arousal
    - labels: 形状与 preds 相同
    返回:
    - 最终平均 CCC（valence和arousal样本级CCC平均值的整体平均）
    """
    # 检查输入有效性
    preds = np.array(preds).astype(float)
    labels = np.array(labels).astype(float)
    assert preds.shape == labels.shape, "预测和标签形状不一致"
    assert preds.ndim == 3 and preds.shape[-1] == 2, "输入应为 (N, T, 2)"

    # 存储每个样本的 Valence 和 Arousal CCC
    valence_cccs = []
    arousal_cccs = []

    # 遍历每个样本 (N=6693)
    for i in range(preds.shape[0]):
        # 提取当前样本的预测和标签
        sample_pred = preds[i]  # (T, 2)
        sample_label = labels[i]  # (T, 2)

        # 分别计算 Valence 和 Arousal 的 CCC
        v_ccc = _compute_single_ccc(sample_pred[:, 0], sample_label[:, 0])
        a_ccc = _compute_single_ccc(sample_pred[:, 1], sample_label[:, 1])

        valence_cccs.append(v_ccc)
        arousal_cccs.append(a_ccc)

    # 计算各维度平均
    mean_v = np.nanmean(valence_cccs)  # 自动忽略 NaN
    mean_a = np.nanmean(arousal_cccs)
    final_ccc = (mean_v + mean_a) / 2

    # return {
    #     "valence_avg_ccc": mean_v,
    #     "arousal_avg_ccc": mean_a,
    #     "final_avg_ccc": final_ccc
    # }
    return final_ccc

def _compute_single_ccc(preds, labels):
    """计算单个序列的 CCC（处理常数情况）"""
    # 检查常数
    if np.all(preds == preds[0]) or np.all(labels == labels[0]):
        return np.nan  # 标记为无效，后续用 nanmean 忽略

    # 计算统计量
    mean_pred = np.mean(preds)
    mean_label = np.mean(labels)
    var_pred = np.var(preds, ddof=0)
    var_label = np.var(labels, ddof=0)
    cov = np.cov(preds, labels, ddof=0)[0, 1]

    # 计算 CCC
    numerator = 2 * cov
    denominator = var_pred + var_label + (mean_pred - mean_label) ** 2 + 1e-10
    return numerator / denominator



# def gain_cv_results(folder_save):
#
#     # find all keys
#     whole_keys = list(folder_save[0].keys())
#
#     cv_acc, cv_fscore, cv_valmse = -100, -100, -100
#     if 'eval_emoacc' in whole_keys:
#         cv_acc = np.mean([epoch_save['eval_emoacc'] for epoch_save in folder_save])
#     if 'eval_emofscore' in whole_keys:
#         cv_fscore = np.mean([epoch_save['eval_emofscore'] for epoch_save in folder_save])
#     if 'eval_valmse' in whole_keys:
#         cv_valmse = np.mean([epoch_save['eval_valmse'] for epoch_save in folder_save])
#
#     # 只显示存在的部分信息 [与test输出是一致的]
#     outputs = []
#     if cv_fscore != -100: outputs.append(f'f1:{cv_fscore:.4f}')
#     if cv_acc    != -100: outputs.append(f'acc:{cv_acc:.4f}')
#     if cv_valmse != -100: outputs.append(f'val:{cv_valmse:.4f}')
#     outputs = "_".join(outputs)
#     return outputs



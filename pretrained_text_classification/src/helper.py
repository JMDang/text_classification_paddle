#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File  :   helper.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   common func
"""

import logging
import numpy as np
# logging.basicConfig(format='%(asctime)s-%(levelname)s - %(message)s', level=logging.INFO)
from sklearn.metrics import  precision_score, recall_score, f1_score



def multi_classify_prf_macro(y_true, y_pre):
    """macro prf
    每个类别看作二分类,每个TP FN FP TN,算prf后平均
    """
    p = precision_score(y_true=y_true, y_pred=y_pre, average="macro")
    r = recall_score(y_true=y_true, y_pred=y_pre, average="macro")
    f = f1_score(y_true=y_true, y_pred=y_pre, average="macro")
    return p,r,f

def multi_classify_prf_micro(y_true, y_pre):
    """micro, prf
    每个类别看作二分类,每个TP FN FP TN,先求和后算一次prf
    """
    p = precision_score(y_true=y_true, y_pred=y_pre, average="micro")
    r = recall_score(y_true=y_true, y_pred=y_pre, average="micro")
    f = f1_score(y_true=y_true, y_pred=y_pre, average="micro")
    return p,r,f

def multi_classify_prf_self(y_true, y_pred):
    """
    除了0类其他类别的PRF
    """
    y_true=np.argmax(y_true,1)
    y_pred = np.argmax(y_pred, 1)
    y_preNotNA = [int(i) for i in np.greater(y_pred, 0)]
    pre_num = np.sum(y_preNotNA)
    y_trueNotNA = [int(i) for i in np.greater(y_true, 0)]
    true_num = np.sum(y_trueNotNA)
    equal_num = np.sum(np.multiply([i for i in np.equal(y_true, y_pred)], y_preNotNA))
    try:
        precision = equal_num / pre_num
        recall = equal_num / true_num
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1

def extract_entity_crf(pre_label, predict_length, label_encoder=None):
    """ 从预测标签抽取实体,标签个数: label_encoder.size() * 2 - 1
    [IN]  pre_label: 模型预测结果
          predict_length: 预测数据真实长度
          label_encoder: 标签编码
    """
    pre_entity = []
    for label, length in zip(pre_label, predict_length):
        item = []
        if length >= len(label):
            length = len(label)
        index = 0
        while index < length:
            if label[index] != 0:
                start = index
                start_type_id = label[index]
                end = index + 1
                while end < length and label[end]==start_type_id + 1:
                    end += 1
                    index += 1
                if start_type_id // 2 + 1 not in label_encoder.id_label_dict:
                    break
                item.append([start - 1, end - 1, label_encoder.inverse_transform(start_type_id // 2 + 1)])
            index += 1
        pre_entity.append(item)
    return pre_entity

def extract_entity_crf_BIO(pre_label, predict_length):
    """ 从预测标签抽取实体,只有三种标签   BIO
    [IN]  pre_label: 模型预测结果
          predict_length: 预测数据真实长度
          label_encoder: 标签编码
    """
    pre_entity = []
    for label, length in zip(pre_label, predict_length):
        item = []
        if length >= len(label):
            length = len(label)
        index = 0
        while index < length:
            if label[index] == 1:
                start = index
                end = index + 1
                while end < length and label[end]==2:
                    end += 1
                item.append([start - 1, end - 1])
            index += 1
        pre_entity.append(item)
    return pre_entity

def extract_entity_dp(pre_label, predict_length, label_encoder):
    """ 从预测标签抽取实体
    [IN]  pre_label: 模型预测结果
          predict_length: 预测数据真实长度
          label_encoder: 标签编码
    """
    pre_entity = []
    for label, length in zip(pre_label, predict_length):
        item = []
        if length >= len(label[0]):
            length = len(label[0])
        for start in range(length):
            if (label[0][start] != 0):
                for end in range(start, length):
                    if (label[1][end] != 0):
                        item.append([start - 1, end, label_encoder.inverse_transform(label[1][end])])
                        break
        pre_entity.append(item)
    return pre_entity








if __name__ == '__main__':
    print(extract_entity_crf([[1, 2, 2, 2, 1, 2, 2]], [7]))

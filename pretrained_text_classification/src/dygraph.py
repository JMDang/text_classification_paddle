#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   dygraph.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   包含训练循环,预测,评估函数,加载模型函数
"""

import os
import sys
import numpy as np
import logging
import time
import copy
import paddle
from data_loader import DataLoader
import helper

logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

def train(model,
          train_data,
          label_encoder,
          dev_data=None,
          epochs=5,
          pretrain_lr=0.001,
          other_lr=0.001,
          weight_decay=0.01,
          batch_size=32,
          max_seq_len=300,
          max_ensure=True,
          model_save_path=None,
          best_model_save_path=None,
          print_step=15,
          acti_fun="softmax",
          threshold=None
        ):
    """动态图训练
    """
    logging.info("train model start")
    train_start_time = time.time()

    model.train()
    if acti_fun == "softmax":
        criterion = paddle.nn.loss.CrossEntropyLoss()
    else:
        # criterion = paddle.nn.loss.BCELoss()#要求输入logits是sigmoid之后的
        criterion = paddle.nn.loss.BCEWithLogitsLoss()#要求输入logits是未经过sigmoid的
    # 统一学习率
    # optimizer = paddle.optimizer.Adam(learning_rate=learning_rate,
    #                                   parameters=model.parameters())
    # 差分学习率
    optimizer = build_optimizer(model, pretrain_lr, other_lr, weight_decay)

    cur_train_step = 0
    best_f1 = 0.0
    for cur_epoch in range(epochs):
        # 每个epoch都shuffle数据以获得最佳训练效果
        np.random.shuffle(train_data)
        train_data_batch = DataLoader.gen_batch_data(train_data,
                                                     batch_size,
                                                     max_seq_len,
                                                     max_ensure,
                                                     with_label=True)
        for cur_train_data, cur_train_length, cur_train_label_mc, cur_train_label_ml in train_data_batch:
            cur_train_step += 1
            # 训练数据转为tensor
            cur_train_data = paddle.to_tensor(cur_train_data)
            cur_train_label_mc = paddle.to_tensor(cur_train_label_mc)
            cur_train_label_ml = paddle.to_tensor(cur_train_label_ml)
            # 生成loss
            logits = model(cur_train_data)
            # softmax多分类,softmax二分类
            if acti_fun == "softmax":
                loss = criterion(logits, cur_train_label_mc)
            # sigmoid二分类
            elif label_encoder.size() == 2:
                cur_train_label_mc = paddle.cast(cur_train_label_mc, "float32")
                logits = paddle.squeeze(logits, 1)
                loss = criterion(logits, cur_train_label_mc)
            # sigmoid多标签
            else:
                cur_train_label_ml = paddle.cast(cur_train_label_ml, "float32")
                loss = criterion(logits, cur_train_label_ml)

            if cur_train_step % print_step == 0:
                speed = cur_train_step / (time.time() - train_start_time)
                logging.info('train epoch %d, step %d: loss %.5f, speed %.2f step/s' % \
                        (cur_epoch, cur_train_step, loss.numpy(), speed))
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # 每轮保存模型
        if model_save_path:
            logging.info("save model at epoch {}".format(cur_epoch))
            start_time = time.time()
            paddle.save(model.state_dict(), model_save_path + "_epoch{}".format(cur_epoch))
            logging.info("cost time: %.4fs" % (time.time() - start_time))

        if dev_data:
            precision, recall, f1 = eval(model=model,
                                         eval_data=dev_data,
                                         label_encoder=label_encoder,
                                         batch_size=batch_size,
                                         max_seq_len=max_seq_len,
                                         max_ensure=max_ensure,
                                         acti_fun=acti_fun,
                                         threshold=threshold)
            logging.info('eval epoch %d, pre %.5f rec %.5f f1 %.5f' % (cur_epoch, precision, recall, f1))

            if best_model_save_path and f1 > best_f1:
                # 如果优于最优acc 则保存为best模型
                logging.info("cur best score, save model at epoch {} as best model".format(cur_epoch))
                start_time = time.time()
                paddle.save(model.state_dict(), best_model_save_path)
                logging.info("cost time: %.4fs" % (time.time() - start_time))
                best_f1 = f1

    logging.info("train model cost time %.4fs" % (time.time() - train_start_time))

def translate(in_list):
    """
    in_list:二纬list
    """
    result = []
    for one_case in in_list:
        one_case_label = []
        for i, item in enumerate(one_case):
            if item == 1:
                one_case_label.append(i)
        result.append(one_case_label)
    return result

def predict(model,
            predict_data,
            label_encoder,
            batch_size=32,
            max_seq_len=300,
            max_ensure=True,
            with_label=False,
            acti_fun="softmax",
            threshold=0.5):
    """ 动态图预测
    [IN]  model:
          predict_data: list[(input1[, input2, ...])], 待预测数据
          label_encoder: 标签编码器
          batch_size: int, 批大小
          max_seq_len: int, 最大长度
          max_ensure: 是否为固定长度
          with_label: 是否包含真是标签
    """

    pre_label = []
    pre_label_name = []
    rea_label = []
    rea_label_name = []

    if with_label:
        predict_data_batch = DataLoader.gen_batch_data(
            predict_data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_ensure=max_ensure,
            with_label=True)
        model.eval()
        for cur_predict_data, cur_predict_length, cur_rea_label_mc, cur_rea_label_ml in predict_data_batch:
            cur_predict_data = paddle.to_tensor(cur_predict_data)
            cur_logits = model(cur_predict_data)
            assert acti_fun in ["softmax", "sigmoid"], f"param acti_fun must in [softmax, sigmoid],yours is {acti_fun}"
            # softmax多分类,softmax二分类
            if acti_fun == "softmax":
                cur_pre_label = paddle.nn.functional.softmax(cur_logits).numpy()
                cur_pre_label = np.argmax(cur_pre_label, axis=-1)
                cur_pre_label_name = [label_encoder.inverse_transform(label_id) for label_id in cur_pre_label]
                cur_rea_label = cur_rea_label_mc

            # sigmoid二分类
            elif label_encoder.size() == 2:
                cur_pre_label = paddle.nn.functional.sigmoid(cur_logits)
                cur_pre_label = paddle.squeeze(cur_pre_label, 1).numpy()
                assert threshold >= 0.0 and threshold <=1.0, f"{threshold} must between [0, 1]"
                cur_pre_label = np.where(cur_pre_label > threshold, 1, 0)
                cur_pre_label_name = [label_encoder.inverse_transform(label_id) for label_id in cur_pre_label]
                cur_rea_label = cur_rea_label_mc
            # sigmoid多标签
            else:
                cur_pre_label = paddle.nn.functional.sigmoid(cur_logits)
                assert threshold >= 0.0 and threshold <= 1.0, f"{threshold} must between [0, 1]"
                cur_pre_label = np.where(cur_pre_label > threshold, 1, 0)
                cur_pre_label_name = [",".join(map(lambda x: label_encoder.inverse_transform(x), one_case_real_label))
                                   for one_case_real_label in translate(cur_pre_label)]
                cur_rea_label = cur_rea_label_ml
            pre_label.extend(cur_pre_label)
            rea_label.extend(cur_rea_label)
            pre_label_name.extend(cur_pre_label_name)
            rea_label_name.extend([",".join(map(lambda x:label_encoder.inverse_transform(x), one_case_real_label))
                                   for one_case_real_label in translate(cur_rea_label_ml)])
        model.train()
        return pre_label, pre_label_name, rea_label, rea_label_name

    else:
        predict_data_batch = DataLoader.gen_batch_data(
            predict_data,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_ensure=max_ensure,
            with_label=False)
        model.eval()
        for cur_predict_data, cur_predict_length in predict_data_batch:
            cur_predict_data = paddle.to_tensor(cur_predict_data)
            cur_predict_length = paddle.to_tensor(cur_predict_length)
            cur_logits = model(cur_predict_data)
            # softmax多分类,softmax二分类
            if acti_fun == "softmax":
                cur_pre_label = paddle.nn.functional.softmax(cur_logits).numpy()
                cur_pre_label = np.argmax(cur_pre_label, axis=-1)
                cur_pre_label_name = [label_encoder.inverse_transform(label_id) for label_id in cur_pre_label]
            # sigmoid二分类
            elif label_encoder.size() == 2:
                cur_pre_label = paddle.nn.functional.sigmoid(cur_logits)
                cur_pre_label = paddle.squeeze(cur_pre_label, 1).numpy()
                assert threshold >= 0.0 and threshold <= 1.0, f"{threshold} must between [0, 1]"
                cur_pre_label = np.where(cur_pre_label > threshold, 1, 0)
                cur_pre_label_name = [label_encoder.inverse_transform(label_id) for label_id in cur_pre_label]
            # sigmoid多标签
            else:
                cur_pre_label = paddle.nn.functional.sigmoid(cur_logits)
                assert threshold >= 0.0 and threshold <= 1.0, f"{threshold} must between [0, 1]"
                cur_pre_label = np.where(cur_pre_label > threshold, 1, 0)
                cur_pre_label_name =  [",".join(map(lambda x: label_encoder.inverse_transform(x), one_case_real_label))
                 for one_case_real_label in translate(cur_pre_label)]

            pre_label.extend(cur_pre_label)
            pre_label_name.extend(cur_pre_label_name)
        model.train()
        return pre_label, pre_label_name

def eval(model,
         eval_data,
         label_encoder,
         batch_size=32,
         max_seq_len=300,
         max_ensure=True,
         acti_fun="softmax",
         threshold=0.5):
    """ eval
    [IN]  model:
          eval_data: list[(input1[, input2, ...], label)], 训练数据
          label_encoder: LabelEncoder, 类别转化工具
          batch_size: int, 批大小
          max_seq_len: int, 最大长度
    [OUT] acc: float, 评估结果
    """
    pre_label, pre_entity, rea_label, rea_entity = predict(model=model,
                                                           predict_data=eval_data,
                                                           label_encoder=label_encoder,
                                                           batch_size=batch_size,
                                                           max_seq_len=max_seq_len,
                                                           max_ensure=max_ensure,
                                                           with_label=True,
                                                           acti_fun=acti_fun,
                                                           threshold=threshold)
    rea_label = np.array(rea_label).flatten()
    pre_label = np.array(pre_label).flatten()
    precision, recall, f1 = helper.multi_classify_prf_macro(rea_label, pre_label)

    return precision, recall, f1

def load_model(model, model_path):
    """ 加载模型
    [in] model: 已构造好的模型结构
         model_path: str, 模型地址
    """
    if os.path.exists(model_path):
        logging.info("load model from {}".format(model_path))
        start_time = time.time()
        sd = paddle.load(model_path)
        model.set_dict(sd)
        logging.info("cost time: %.4fs" % (time.time() - start_time))
    else:
        logging.info("cannot find model file: {}".format(model_path))

# 差分学习率
def build_optimizer(model, pretrain_model_lr, other_lr, weight_decay):

    # 差分学习率
    no_decay = ["bias", "layer_norm.weight"]
    pretrain_param_optimizer = []
    other_param_optimizer = []

    for name, para in model.named_parameters():
        space = name.split('.')
        if space[0] == 'ernie':
            pretrain_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # pretrain_models
        {"params": [p for n, p in pretrain_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': pretrain_model_lr},
        {"params": [p for n, p in pretrain_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': pretrain_model_lr},

        # other module
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': other_lr},
    ]

    optimizer = paddle.optimizer.Adam(learning_rate = pretrain_model_lr,
                                      parameters = optimizer_grouped_parameters)

    return optimizer


if __name__ == "__main__":
    pass



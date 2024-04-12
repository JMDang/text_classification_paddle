#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   train.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   train入口
"""

import os
import sys
import logging
import configparser
import numpy as np
import paddle
import dygraph
from data_loader import DataLoader
from label_encoder import LabelEncoder
from models.classify_model import BoWModel, LSTMModel, BiLSTMAtt,GRUModel,BiGRUAtt,CNNModel
from layers.self_layers import WORD_ATT_V1, WORD_ATT_V2, SelfAttention, SelfInteractiveAttention

logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

class Train:
    """模型训练
    """
    def __init__(self, train_conf_path):
        self.train_conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.train_conf.read(train_conf_path)

        Train.check_dir(self.train_conf["DEFAULT"]["input_dir"])
        Train.check_dir(self.train_conf["DEFAULT"]["output_dir"])
        Train.check_dir(self.train_conf["DEFAULT"]["model_dir"])

        self.label_encoder = LabelEncoder(label_id_info=self.train_conf["DATA"]["label_encoder_path"], 
                                    isFile=True)
        for label_id, label_name in sorted(self.label_encoder.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))

    
    def run(self):
        """执行入口
        """
        self.data_set = DataLoader(label_encoder=self.label_encoder,
                                   vocab_path=self.train_conf["DATA"]["vocab_path"])
        self.data_set.gen_data(
            train_data_dir=self.train_conf["DATA"]["train_data_path"],
            dev_data_dir=self.train_conf["DATA"]["dev_data_path"] if \
                self.train_conf["DATA"]["dev_data_path"] != "None" else None,
            test_data_dir=self.train_conf["DATA"]["test_data_path"] if \
                self.train_conf["DATA"]["test_data_path"] != "None" else None
        )

        if self.train_conf["RUN"].getboolean("train_or_predict"):
            Train.train(self.train_conf,
                        self.data_set,
                        self.label_encoder)

        if self.train_conf["RUN"].getboolean("to_static"):
            self.to_static(train_conf=self.train_conf,
                                 label_encoder=self.label_encoder,
                                 )
            logging.info("[IMPORTANT] model to static")
    
    @staticmethod
    def check_dir(dir_address):
        """检测目录是否存在
            1. 若不存在则新建
            2. 若存在但不是文件夹，则报错
            3. 若存在且是文件夹则返回
        """
        if not os.path.isdir(dir_address):
            if os.path.exists(dir_address):
                raise ValueError("specified address is not a directory: %s" % dir_address)
            else:
                logging.info("create directory: %s" % dir_address)
                os.makedirs(dir_address)
    
    @staticmethod
    def train(train_conf,
               data_set,
               label_encoder
               ):
        """train微调
        """

        model_type = train_conf["model"]["model_type"]
        num_classes = 1 if  train_conf["model"]["acti_fun"] == "sigmoid" \
                            and label_encoder.size() == 2 else label_encoder.size()
        use_w2v_emb = train_conf["model"].getboolean("use_w2v_emb")
        if model_type == "BoWModel":
            model = BoWModel(num_classes=num_classes,
                             use_w2v_emb=use_w2v_emb,
                             extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "LSTMModel":
            model = LSTMModel(num_classes=num_classes,
                              use_w2v_emb=use_w2v_emb,
                              extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "BiLSTMAtt":
            model = BiLSTMAtt(attention_layer=SelfAttention(hidden_size=512),
                              num_classes=num_classes,
                              use_w2v_emb=use_w2v_emb,
                              extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "GRUModel":
            model = GRUModel(num_classes=num_classes,
                              use_w2v_emb=use_w2v_emb,
                              extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "BiGRUAtt":
            model = BiGRUAtt(attention_layer=WORD_ATT_V1(fea_size=512, attention_size=256),
                              num_classes=num_classes,
                              use_w2v_emb=use_w2v_emb,
                              extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "CNNModel":
            model = CNNModel(num_classes=num_classes,
                             use_w2v_emb=use_w2v_emb,
                             extended_vocab_path=train_conf["DATA"]["vocab_path"])
        else:
            raise ValueError(f"unknown model_type: {model_type} model_type, \
            model_type must in  [BoWModel, LSTMModel, BiLSTMAtt, GRUModel, BiGRUAtt, CNNModel")



        dygraph.load_model(model, train_conf["MODEL_FILE"]["model_best_path"])

        dygraph.train(model,
                      train_data=data_set.train_data,
                      label_encoder=label_encoder,
                      dev_data=data_set.dev_data,
                      epochs=train_conf["model"].getint("epoch"),
                      learning_rate=train_conf["model"].getfloat("learning_rate"),
                      batch_size=train_conf["model"].getint("batch_size"),
                      max_seq_len=train_conf["model"].getint("max_seq_len"),
                      model_save_path=train_conf["MODEL_FILE"]["model_path"],
                      best_model_save_path=train_conf["MODEL_FILE"]["model_best_path"],
                      print_step=train_conf["model"].getint("print_step"),
                      acti_fun=train_conf["model"]["acti_fun"],
                      threshold=train_conf["model"].getfloat("threshold"))

    @staticmethod
    def to_static(train_conf,
                  label_encoder):
        """模型转静态图模型文件
        """
        model_type = train_conf["model"]["model_type"]
        num_classes = 1 if train_conf["model"]["acti_fun"] == "sigmoid" \
                           and label_encoder.size() == 2 else label_encoder.size()
        use_w2v_emb = train_conf["model"].getboolean("use_w2v_emb")
        if model_type == "BoWModel":
            model = BoWModel(num_classes=num_classes,
                             use_w2v_emb=use_w2v_emb,
                             extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "LSTMModel":
            model = LSTMModel(num_classes=num_classes,
                              use_w2v_emb=use_w2v_emb,
                              extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "BiLSTMAtt":
            model = BiLSTMAtt(attention_layer=SelfAttention(hidden_size=256),
                              num_classes=num_classes,
                              use_w2v_emb=use_w2v_emb,
                              extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "GRUModel":
            model = GRUModel(num_classes=num_classes,
                             use_w2v_emb=use_w2v_emb,
                             extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "BiGRUAtt":
            model = BiGRUAtt(attention_layer=WORD_ATT_V1(fea_size=512, attention_size=256),
                             num_classes=num_classes,
                             use_w2v_emb=use_w2v_emb,
                             extended_vocab_path=train_conf["DATA"]["vocab_path"])
        elif model_type == "CNNModel":
            model = CNNModel(num_classes=num_classes,
                             use_w2v_emb=use_w2v_emb,
                             extended_vocab_path=train_conf["DATA"]["vocab_path"])
        else:
            raise ValueError(f"unknown model_type: {model_type} model_type, \
                    model_type must in  [BoWModel, LSTMModel, BiLSTMAtt, GRUModel, BiGRUAtt, CNNModel")

        dygraph.load_model(model, train_conf["MODEL_FILE"]["model_best_path"])

        model.eval()

        model = paddle.jit.to_static(model,
                                    input_spec=[paddle.static.InputSpec(shape=[None, train_conf["model"].getint("max_seq_len")],dtype="int32"),
                                                None])

        paddle.jit.save(model, train_conf["MODEL_FILE"]["model_static_path"])
        logging.info("save static model to {}".format(train_conf["MODEL_FILE"]["model_static_path"]))

if __name__ == "__main__":
    Train(train_conf_path=sys.argv[1]).run()




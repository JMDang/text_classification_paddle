#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   predict.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   predict入口
"""

import os
import sys
import json
import logging
import configparser
import numpy as np
import paddle
import dygraph
from data_loader import DataLoader
from label_encoder import LabelEncoder
from models.classify_model import BoWModel, LSTMModel, BiLSTMAtt,GRUModel,BiGRUAtt,CNNModel
from paddlenlp.embeddings import TokenEmbedding,list_embedding_name
from layers.self_layers import WORD_ATT_V1, WORD_ATT_V2, SelfAttention, SelfInteractiveAttention
import jieba

class Predict:
    """模型预测
    """
    def __init__(self, predict_conf_path):
        self.predict_conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.predict_conf.read(predict_conf_path)

        self.label_encoder = LabelEncoder(label_id_info=self.predict_conf["DATA"]["label_encoder_path"],
                                    isFile=True)
        for label_id, label_name in sorted(self.label_encoder.id_label_dict.items(), key=lambda x:x[0]):
            logging.info("%d: %s" % (label_id, label_name))

        self.vocab = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                              extended_vocab_path=self.predict_conf["DATA"]["vocab_path"],
                                              unknown_token="[UNK]"
                                              ).vocab
        self.vocab_level = DataLoader.get_vocab_level(self.vocab.token_to_idx)
    def run(self):
        """执行入口
        """
        if self.predict_conf["RUN"].getboolean("train_or_predict"):
            label_encoder = self.label_encoder

            model_type = self.predict_conf["model"]["model_type"]
            num_classes = 1 if self.predict_conf["model"]["acti_fun"] == "sigmoid" \
                               and label_encoder.size() == 2 else label_encoder.size()
            use_w2v_emb = self.predict_conf["model"].getboolean("use_w2v_emb")
            if model_type == "BoWModel":
                model_predict = BoWModel(num_classes=num_classes,
                                 use_w2v_emb=use_w2v_emb,
                                 extended_vocab_path=self.predict_conf["DATA"]["vocab_path"])
            elif model_type == "LSTMModel":
                model_predict = LSTMModel(num_classes=num_classes,
                                  use_w2v_emb=use_w2v_emb,
                                  extended_vocab_path=self.predict_conf["DATA"]["vocab_path"])
            elif model_type == "BiLSTMAtt":
                model_predict = BiLSTMAtt(attention_layer=SelfAttention(hidden_size=256),
                                  num_classes=num_classes,
                                  use_w2v_emb=use_w2v_emb,
                                  extended_vocab_path=self.predict_conf["DATA"]["vocab_path"])
            elif model_type == "GRUModel":
                model_predict = GRUModel(num_classes=num_classes,
                                 use_w2v_emb=use_w2v_emb,
                                 extended_vocab_path=self.predict_conf["DATA"]["vocab_path"])
            elif model_type == "BiGRUAtt":
                model_predict = BiGRUAtt(attention_layer=WORD_ATT_V1(fea_size=512, attention_size=256),
                                 num_classes=num_classes,
                                 use_w2v_emb=use_w2v_emb,
                                 extended_vocab_path=self.predict_conf["DATA"]["vocab_path"])
            elif model_type == "CNNModel":
                model_predict = CNNModel(num_classes=num_classes,
                                 use_w2v_emb=use_w2v_emb,
                                 extended_vocab_path=self.predict_conf["DATA"]["vocab_path"])
            else:
                raise ValueError(f"unknown model_type: {model_type} model_type, \
                                model_type must in  [BoWModel, LSTMModel, BiLSTMAtt, GRUModel, BiGRUAtt, CNNModel")

            dygraph.load_model(model_predict, self.predict_conf["MODEL_FILE"]["model_best_path"] + ".pdparams")

            predict_data = []
            text_list = []
            length_list = []
            origin_texts = []
            mark = 0
            tmp_d = {}
            for line in sys.stdin:
                mark = mark + 1
                cols = line.strip("\n").split("\t")
                origin_text = cols[0]
                if self.vocab_level == "word":
                    text = self.vocab.to_indices(list(jieba.cut(origin_text)))
                else:
                     text = self.vocab.to_indices([ch for ch in origin_text])
                text = self.vocab.to_indices(list(origin_text))
                text_list.append(text)
                length_list.append(len(text))
                origin_texts.append(origin_text)
                tmp_d[origin_text] = cols

                if mark == 32:
                    predict_data = list(zip(text_list, length_list))
                    pre_label, pre_label_name = dygraph.predict(model=model_predict,
                                                            predict_data=predict_data,
                                                            label_encoder=label_encoder,
                                                            batch_size=self.predict_conf["model"].getint("batch_size"),
                                                            max_seq_len=self.predict_conf["model"].getint("max_seq_len"),
                                                            max_ensure=True,
                                                            with_label=False,
                                                            acti_fun=self.predict_conf["model"]["acti_fun"],
                                                            threshold=self.predict_conf["model"].getfloat("threshold"))
                    for origin_text, one_pre_label_name in zip(origin_texts, pre_label_name):
                        print(origin_text,"\t", "\t".join(tmp_d[origin_text][1:]), "\t", one_pre_label_name)
                    predict_data = []
                    text_list = []
                    length_list = []
                    origin_texts = []
                    mark = 0
            if mark != 0:
                predict_data = list(zip(text_list, length_list))
                pre_label, pre_label_name = dygraph.predict(model=model_predict,
                                                        predict_data=predict_data,
                                                        label_encoder=label_encoder,
                                                        batch_size=self.predict_conf["model"].getint("batch_size"),
                                                        max_seq_len=self.predict_conf["model"].getint("max_seq_len"),
                                                        max_ensure=True,
                                                        with_label=False,
                                                        acti_fun=self.predict_conf["model"]["acti_fun"],
                                                        threshold=self.predict_conf["model"].getfloat("threshold"))

                for origin_text, one_pre_label_name in zip(origin_texts, pre_label_name):
                    print(origin_text, "\t", "\t".join(tmp_d[origin_text][1:]), "\t", one_pre_label_name)

if __name__ == "__main__":
    Predict(predict_conf_path=sys.argv[1]).run()

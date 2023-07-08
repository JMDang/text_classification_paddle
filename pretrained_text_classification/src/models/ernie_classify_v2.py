#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   ernie_classify_v2.py
Author:   dangjinming(776039904@qq.com)
Date  :   2022/3/16
Desc  :   ernie_crf_v2
"""

import sys
import os
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErnieModel


import logging
logging.basicConfig(
    format='"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    level=logging.INFO)

class ErnieForSequenceClassificationV2(nn.Layer):
    def __init__(self, ernie, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])

        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    self.num_classes)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, pooled_output = self.ernie(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



if __name__ == "__main__":
    pretrain_ernie = ErnieModel.from_pretrained("ernie-1.0") #ernie未进行下游任务适配

    model = ErnieForSequenceClassificationV2(pretrain_ernie, num_classes=2)
    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    inputs = tokenizer("隔壁王老师在哪里!")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    logits = model(**inputs)
    print(logits)

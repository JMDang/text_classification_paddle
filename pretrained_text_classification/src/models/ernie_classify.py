#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   ernie_classify.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   ernie_2fc
"""

import sys
import paddle
import paddle.nn as nn
from paddlenlp.transformers import ErniePretrainedModel
from paddle.nn import  BatchNorm, Dropout,Flatten

class ErnieForSequenceClassification(ErniePretrainedModel):
    
    """
    此处实现为仿照paddlenlp/transformers/ernie/modeling.py中的ErnieForSequenceClassification进行自定义类实现
    paddlenlp2.2
        ErnieForSequenceClassification.from_pretrained('ernie-1.0', num_classes=7, id2label=id2label)
        参数直接原封不动透传给model，如果提前内置(ErnieForSequenceClassification.__init__).parameters中)则改变内置值，否则不生效(不报错)，但可自定义类绑定
    paddlenlp2.4.6之后更高版本
        ErnieForSequenceClassification.from_pretrained('ernie-1.0', num_classes=7, id2label=id2label)
        参数直接原封不动透传给model，如果提前内置(ErnieConfig->PretrainedConfig中)则改变内置值，否则不生效(会报错)，但可自定义类绑定
    """
    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])
        self.bn = BatchNorm(num_channels=1, data_layout="NHWC")
        self.classifier_hid = nn.Linear(self.ernie.config["hidden_size"],
                                     self.ernie.config["hidden_size"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    self.num_classes)


        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        sequence_output, pooled_output = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        # sequence_output = self.dropout(sequence_output)
        # sequence_output = self.bn(paddle.unsqueeze(sequence_output, axis=-1))
        # sequence_output = self.classifier_hid(sequence_output)
        # sequence_output = self.dropout(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

if __name__ == "__main__":
    ernie = ErnieForSequenceClassification.from_pretrained("ernie-1.0",num_classes=2)
    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('ernie-1.0')

    inputs = tokenizer("隔壁王老师在哪里!")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    logits = ernie(**inputs)
    print(logits)



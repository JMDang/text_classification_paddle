#!/usr/bin/env python
# -*- utf-8 -*-
"""
File  :   classify_model.py
Author:   dangjinming(jmdang777@qq.com)
Date  :   2022/3/16
Desc  :   classify_model
"""

import sys
import paddle
import paddlenlp
import paddle.nn as nn
from paddlenlp.embeddings import TokenEmbedding,list_embedding_name
from paddle.nn import  BatchNorm, Dropout,Flatten
from layers.self_layers import LAST_STEP, WORD_ATT_V1, WORD_ATT_V2, SelfAttention, SelfInteractiveAttention

def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, "r", encoding="utf-8") as fin:
        for line in fin:
            key = line.strip("\n")
            vocab[key] = i
            i += 1
    return vocab


class BoWModel(nn.Layer):
    """
    BoWModel
    """
    def __init__(self,
                 num_classes,
                 emb_size=300,
                 hidden_size=256,
                 fc_hidden_size=96,
                 word_num=30000,
                 use_w2v_emb=False,
                 extended_vocab_path=None,
                 padding_idx=0):

        super().__init__()
        self.padding_idx = padding_idx
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
            self.padding_idx = self.word_emb._word_to_idx["[PAD]"]
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)

        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, inputs, true_lengths=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.word_emb(inputs)
        # Shape: (batch_size, embedding_dim)
        bow_summed = embedded_text.sum(axis=1)
        encoded_text = paddle.tanh(bow_summed)
        # Shape: (batch_size, hidden_size)
        fc1_out = paddle.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_size)
        fc2_out = paddle.tanh(self.fc2(fc1_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc2_out)
        return logits

class LSTMModel(nn.Layer):
    def __init__(
        self,
        num_classes,
        emb_size=300,
        lstm_hidden_size=256,
        fc_hidden_size=96,
        lstm_layers=2,
        word_num=30000,
        use_w2v_emb=False,
        extended_vocab_path=None,
        padding_idx=0,
        pooling_type="mean"
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.pooling_type = pooling_type
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
            self.padding_idx = self.word_emb._word_to_idx["[PAD]"]
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            direction="forward")

        self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, inputs, true_lengths=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.word_emb(inputs)
        encoded_text, (last_hidden, last_cell) = self.lstm(embedded_text,sequence_length=true_lengths)
        if self.pooling_type == 'sum':
            encoded_text_pool = paddle.sum(encoded_text, axis=1)
        elif self.pooling_type == 'max':
            encoded_text_pool = paddle.max(encoded_text, axis=1)
        elif self.pooling_type == 'mean':
            encoded_text_pool = paddle.mean(encoded_text, axis=1)
        else:
            raise RuntimeError(
                "Unexpected pooling type %s ."
                "Pooling type must be one of sum, max and mean." %
                self.pooling_type)

        fc_out = paddle.tanh(self.fc(encoded_text_pool))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits

class BiLSTMAtt(nn.Layer):
    def __init__(
        self,
        attention_layer,
        num_classes,
        emb_size=300,
        lstm_hidden_size=256,
        fc_hidden_size=96,
        lstm_layers=2,
        word_num=30000,
        use_w2v_emb=False,
        extended_vocab_path=None,
        padding_idx=0
    ):
        super().__init__()
        self.padding_idx = padding_idx
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
            self.padding_idx = self.word_emb._word_to_idx["[PAD]"]
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)

        self.bilstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            direction="bidirectional",
        )

        self.attention = attention_layer
        if isinstance(attention_layer, SelfAttention):
            self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        elif isinstance(attention_layer, SelfInteractiveAttention) or \
                isinstance(attention_layer, WORD_ATT_V1) or \
            isinstance(attention_layer, WORD_ATT_V2):
            self.fc = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        else:
            raise RuntimeError("Unknown attention type %s." % attention_layer.__class__.__name__)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, inputs, true_lengths=None):
        mask = inputs != self.padding_idx
        embedded_text = self.word_emb(inputs)
        # Encode text, shape: (b, s, num_directions * hidden_size)
        encoded_text, (last_hidden, last_cell) = self.bilstm(embedded_text, sequence_length=true_lengths)
        # Shape: (b, lstm_hidden_size)
        hidden, att_weights = self.attention(encoded_text, mask)
        # Shape: (b, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(hidden))
        # Shape: (b, num_classes)
        logits = self.output_layer(fc_out)
        return logits

class GRUModel(nn.Layer):
    def __init__(
        self,
        num_classes,
        emb_size=300,
        lstm_hidden_size=256,
        fc_hidden_size=96,
        lstm_layers=2,
        word_num=30000,
        use_w2v_emb=False,
        extended_vocab_path=None,
        padding_idx=0,
        pooling_type="mean"
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.pooling_type = pooling_type
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
            self.padding_idx = self.word_emb._word_to_idx["[PAD]"]
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            direction="forward")

        self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, inputs, true_lengths=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.word_emb(inputs)
        encoded_text, gru_h = self.gru(embedded_text, sequence_length=true_lengths)
        if self.pooling_type == 'sum':
            encoded_text_pool = paddle.sum(encoded_text, axis=1)
        elif self.pooling_type == 'max':
            encoded_text_pool = paddle.max(encoded_text, axis=1)
        elif self.pooling_type == 'mean':
            encoded_text_pool = paddle.mean(encoded_text, axis=1)
        else:
            raise RuntimeError(
                "Unexpected pooling type %s ."
                "Pooling type must be one of sum, max and mean." %
                self.pooling_type)

        fc_out = paddle.tanh(self.fc(encoded_text_pool))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits

class BiGRUAtt(nn.Layer):
    def __init__(
        self,
        attention_layer,
        num_classes,
        emb_size=300,
        lstm_hidden_size=256,
        fc_hidden_size=96,
        lstm_layers=2,
        word_num=30000,
        use_w2v_emb=False,
        extended_vocab_path=None,
        padding_idx=0
    ):
        super().__init__()
        self.padding_idx = padding_idx
        if use_w2v_emb:
            '''
            embedding_name:预训练的词向量,list_embedding_name()可看到所有paddlenlp提供的预训练词向量
            extended_vocab_path:可默认,也可用自定义vocab,只是改变id和word的映射关系,不会改变word对应的预训练的向量,id->word->embedings
                                使用默认的词汇,[PAD],[UNK]的idx根据预训练的词包而定
                                个人更喜欢自定义[PAD]->0,[UNK]->1,[NUM]->2....others的映射关系,这样padding的都是0
            '''
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
            self.padding_idx = self.word_emb._word_to_idx["[PAD]"]
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)

        self.bigru = nn.GRU(
            input_size=emb_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            direction="bidirectional",
        )

        self.attention = attention_layer
        if isinstance(attention_layer, SelfAttention):
            self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        elif isinstance(attention_layer, SelfInteractiveAttention) or \
                isinstance(attention_layer, WORD_ATT_V1) or \
            isinstance(attention_layer, WORD_ATT_V2):
            self.fc = nn.Linear(lstm_hidden_size * 2, fc_hidden_size)
        else:
            raise RuntimeError("Unknown attention type %s." % attention_layer.__class__.__name__)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, inputs, true_lengths=None):
        mask = inputs != self.padding_idx
        embedded_text = self.word_emb(inputs)
        # Encode text, shape: (b, s, num_directions * hidden_size)
        encoded_text, gru_h = self.bigru(embedded_text, sequence_length=true_lengths)
        # Shape: (b, lstm_hidden_size)
        hidden, att_weights = self.attention(encoded_text, mask)
        # Shape: (b, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(hidden))
        # Shape: (b, num_classes)
        logits = self.output_layer(fc_out)
        return logits

class CNNModel(nn.Layer):
    """
    cnn
    """

    def __init__(
        self,
        num_classes,
        emb_size=300,
        padding_idx=0,
        num_filter=128,
        ngram_filter_sizes=(3,4,5,),
        fc_hidden_size=96,
        word_num=30000,
        use_w2v_emb=False,
        extended_vocab_path=None
    ):
        super().__init__()
        self.padding_idx = padding_idx
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                                           extended_vocab_path=extended_vocab_path,
                                           unknown_token="[UNK]")
            emb_size = self.word_emb.embedding_dim
            self.padding_idx = self.word_emb._word_to_idx["[PAD]"]
        else:
            if extended_vocab_path:
                vocab = load_dict(extended_vocab_path)
                self.word_emb = nn.Embedding(len(vocab), emb_size)
            else:
                self.word_emb = nn.Embedding(word_num, emb_size)

        self.convs = paddle.nn.LayerList([
            nn.Conv2D(
                in_channels=1,
                out_channels=num_filter,
                kernel_size=(i, emb_size),
            data_format="NHWC") for i in ngram_filter_sizes
        ])
        self.fc_hid = nn.Linear(num_filter * len(ngram_filter_sizes), fc_hidden_size)
        self.fc = nn.Linear(fc_hidden_size, num_classes)


    def forward(self, inputs, true_lengths=None):
        # Shape: (b, s, e)
        embedded_text = self.word_emb(inputs)
        # Shape: (b, s, e, 1) = (N, H, W, C)
        embedded_text = embedded_text.unsqueeze(3)
        # Shape: (b, s_, 1, num_filter) -> (b, s_, num_filter)
        convs_out = [
            paddle.tanh(conv(embedded_text)).squeeze(2) for conv in self.convs]
        maxpool_out = [
            paddle.nn.functional.adaptive_max_pool1d(one_conv.transpose([0,2,1]), output_size=1).squeeze(2)
            for one_conv in convs_out
        ]
        #shape (b, num_filter*len(ngram_filter_sizes))
        encoder_out = paddle.concat(maxpool_out, axis=1)
        encoder_out = paddle.tanh(encoder_out)
        # Shape: (b, fc_hidden_size)
        fc_out = self.fc_hid(encoder_out)
        # Shape: (b, num_classes)
        logits = self.fc(fc_out)
        return logits


if __name__ == "__main__":
    # model =  BiLSTMAtt(WORD_ATT_V1(512, 200), 3,use_w2v_emb=True, extended_vocab_path="../../input/vocab.txt",)
    # model = BoWModel(num_classes=3, use_w2v_emb=True, extended_vocab_path="../../input/vocab.txt")
    # model = LSTMModel(num_classes=3, use_w2v_emb=True, extended_vocab_path="../../input/vocab.txt")
    # model = GRUModel(num_classes=3, use_w2v_emb=True, extended_vocab_path="../../input/vocab.txt")
    # model = BiGRUAtt(WORD_ATT_V1(512, 200), 3, use_w2v_emb=True, extended_vocab_path="../../input/vocab.txt", )
    model = CNNModel(num_classes=3, use_w2v_emb=True, extended_vocab_path="../../input/vocab.txt")


    emb = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300",
                         extended_vocab_path="../../input/vocab.txt",
                         unknown_token="[UNK]")

    inputs = emb.vocab.to_indices(list("隔壁王老师在哪里!"))
    inputs = paddle.to_tensor([inputs])
    true_lengths = paddle.to_tensor([5])
    print(inputs)
    print(true_lengths)
    logits = model(inputs=inputs, true_lengths=true_lengths)
    print(logits)

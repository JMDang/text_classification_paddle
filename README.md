# text_classification_paddle

### 代码说明

paddle框架实现BoWModel, LSTMModel, BiLSTMAtt,GRUModel,BiGRUAtt,CNNModel模型,包括查分学习率,softmax多分类任务,sigmoid二分类任务。均可在[text_classification_pretrain_paddle/config/train_conf.ini](https://github.com/JMDang/text_classification_paddle/blob/main/text_classification_paddle/config/train_conf.ini)进行配置。具体使用那种模型在config中可以配置，一目了然。

### 运行步骤

1.**标签配置:**在[text_classification_paddle/input/label.txt](https://github.com/JMDang/text_classification_paddle/blob/main/text_classification_paddle/input/label.txt)中进行标签配置,格式参考Demo,如果是sigmoid的二分类建议labelid为0的是负样本,1的是正样本。

2.**数据准备:**在[text_classification_paddle/input/train_data/train.txt](https://github.com/JMDang/text_classification_paddle/blob/main/text_classification_paddle/input/train_data/train.txt)中按照demo格式放入待训练的数据，两列，第一列为需要分类的文本,第二列为labelname(类别需在${project_dir}/input/label.txt配置)。同理，可在dev_data和test_data增加验证和测试数据

3.**环境准备:**按照requirments.txt安装相应的包即可，修改[text_classification_paddle/env.sh](https://github.com/JMDang/text_classification_paddle/blob/main/text_classification_paddle/env.sh)配置cuda位置和使用的gpu卡，默认0卡。然后终端执行 `source env.sh `

4.**训练模型：**`python3 src/train.py config/train_conf.ini`模型会保存在text_classification_paddle/model/dygraph/(动态图模型)和text_classification_paddle/model/dygraph/(静态图模型用于推理部署) 文件夹中(脚本自动创建文件夹)

5.**预测模型：**`cat input/test_data/test.txt | python3 src/predict.py config/train_conf.ini` 预测结果会直接打印到终端，可自行重定向到指定文件。

**其他:**如果遇到任何问题，可以给本人邮箱776039904@qq.com发邮件，看到都会回复。


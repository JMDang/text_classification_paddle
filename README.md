# text_classification_paddle

### 代码说明

${project}/pretrained_text_classification

paddle框架写的基于ERNIE的text classification。采用ernie+fc的方式训练,包括差分学习率,softmax多分类任务(acti_fun: softmax),sigmoid二分类任务(acti_fun: sigmoid,且训练数据标签种类只能是2个,在), sigmoid多标签任务(acti_fun: sigmoid,且训练数据标签种类超过2个)。均可在${project}/pretrained_text_classification/config/train_conf.ini进行配置。具体使用那种模型在config中可以配置，一目了然。

${project}/text_classification

paddle框架实现**BoWModel, LSTMModel, BiLSTMAtt,GRUModel,BiGRUAtt,CNNModel**模型,包括差分学习率,**softmax多分类任务**(acti_fun: softmax),**sigmoid二分类任务**(acti_fun: sigmoid,且训练数据标签种类只能是2个,在), **sigmoid多标签任务**(acti_fun: sigmoid,且训练数据标签种类超过2个)。均可在${project}/text_classification/config/train_conf.ini进行配置。具体使用那种模型在config中可以配置，一目了然。


### 运行步骤
1.**标签配置:**在/input/label.txt中进行标签配置,格式参考Demo,如果是sigmoid的二分类建议labelid为0的是负样本,1的是正样本。

2.**数据准备:**在/input/train_data/train.txt中按照demo格式放入待训练的数据，两列，第一列为需要分类的文本,第二列为labelname(类别需在${project_dir}/input/label.txt配置)。同理，可在dev_data和test_data增加验证和测试数据

3.**环境准备:**按照requirments.txt安装相应的包即可，修改/env.sh配置cuda位置和使用的gpu卡，默认0卡。然后终端执行 `source env.sh `

4.**训练模型：**`python3 src/train.py config/train_conf.ini`模型会保存在text_classification_paddle/model/dygraph/(动态图模型)和text_classification_paddle/model/dygraph/(静态图模型用于推理部署) 文件夹中(脚本自动创建文件夹)

5.**预测模型：**`cat input/test_data/test.txt | python3 src/predict.py config/train_conf.ini` 预测结果会直接打印到终端，可自行重定向到指定文件。

**其他:** 友情提供多分类,多标签,二分类标数据集供学习交流
链接：https://pan.baidu.com/s/1A9VEjvgcOGznTeSPaRGrIQ?pwd=3J36 
提取码：3J36 

如果遇到任何问题，可以给本人邮箱jmdang777@qq.com发邮件，看到都会回复。


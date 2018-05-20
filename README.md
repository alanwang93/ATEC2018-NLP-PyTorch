# ATEC2018-NLP

## Run the model
### Train
0. `mkdir log & mkdir checkpoints` 创建文件夹
1. `bash process.sh <config>` 预处理数据（根据`config.py`中定义的名为`<config>`的设定来处理）
2. `python2 train.py --config <config>` 训练名为`<config>`的模型

Other options:

* `--disable_cuda`, 禁止使用GPU, 如果没有GPU会自动使用CPU, 不需要加这个
* `--cuda_num <n>`, 使用编号为n的GPU

## TODO
1. 增加一个简单的模型，把整个pipeline完善
2. 寻找一些预训练好的中文word embedding, 或者找一个大的语料库进行训练
3. 按照Extractor/TokenExtractor的格式写一些提取其他feature的类

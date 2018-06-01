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
1. 继续完善pipeline
2. 找一个大的语料库训练Char Word2Vec


## Notes
1. 在计算score的时候，给pos类的权重越大，threshold（大于这个值认为label是1）也需要越大。

## Results
2018-05-29: 0.6188, siamese_char_best
2018-05-30: 0.5808, att_siamese_default_best
2018-05-30: 0.6142, att_siamese_small_embed_best
2018-05-31: 0.6265, siamese_large_best

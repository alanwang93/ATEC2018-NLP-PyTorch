# ATEC2018-NLP

[TOC]



## 资料

### Blogs & Other sources

1. [文本分类实战系列（一）：特征工程](http://www.jeyzhang.com/text-classification-in-action.html)
2. [Semantic Textual Similarity Wiki](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page)
3. [Github - Paraphrase Identification](https://github.com/wasiahmad/paraphrase_identification_task)
4. [Paraphrase Identification Models in Tensorflow](https://blog.nelsonliu.me/2017/05/20/paraphrase-identification-in-tensorflow/)
   * 作者实现了3个论文的模型，有一些心得
5. 



### Quora Question Pairs Challenge

1. [Solutions](https://www.kaggle.com/c/quora-question-pairs/discussion/34325)
2. [1st solution](https://www.kaggle.com/c/quora-question-pairs/discussion/34355)
3. [Interesting features&models](https://www.kaggle.com/c/quora-question-pairs/discussion/32819)
4. [Importance of cleaning data](https://www.kaggle.com/currie32/the-importance-of-cleaning-text)
5. [7th solution, DL method](https://www.kaggle.com/c/quora-question-pairs/discussion/34697)
6. [24th solution, Generalization](https://www.kaggle.com/c/quora-question-pairs/discussion/34534)
7. [8th solution](https://www.kaggle.com/c/quora-question-pairs/discussion/34371)
8. [Zhihu](https://www.zhihu.com/question/49424474)
9. [Zhihu 2](https://zhuanlan.zhihu.com/p/35093355)
10. [文档相似度](https://www.zhihu.com/question/33952003)
11. 

### Papers

1. [2016] Text Matching as Image Recognition
   - interation focused model，CNN效果不是很好，或者说主要是难以融合额外feature
2. [2014] Convolutional Neural Network Architectures for Matching Natural Language Sentences
   * CNN系列的开始
3. [2016] Wide & Deep Learning for Recommender Systems
   - 结合深度学习和传统LR等模型
4. [2016] **Siamese Recurrent Architectures for Learning Sentence Similarity**
   * 基础
5. [2016] **Learning Natural Language Inference using Bidirectional LSTM model and Inner-Attention**
6. [2016] Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks
   * 比较复杂，不建议
7. [2016] Bilateral Multi-Perspective Matching for Natural Language Sentences
   * 可以看看
8. [2015] Applying Deep Learning to Answer Selection: A Study and An Open Task
   - 很多结构的实验，不是很有用
9. [2016] Learning Text Similarity with Siamese Recurrent Networks
   - 貌似不是很管用
10. [2016] Attentive Pooling Networks
   - 用max pooling 做attention，貌似对paraphrase identification不是很有用，因为有很多重复的词，他们的attention会比较大，但实际上这些词对于判别并不是很有用。
11. [2016] ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
    * 不建议
12. [2016] A Chinese Question Answering Approach Integrating Count-based and Embedding-based Features
    * 可以看看中文特征提取
13. [2018] **Character-based Neural Networks for Sentence Pair Modeling**
14. [2016] **A decomposable attention model for natural language inference**
    * 重点看，Kaggle比赛多个选手用了这个模型
15. [2017] **Neural Paraphrase Identification of Questions with Noisy Pretraining**
    * Google的，值得一看

### Paper notes

1. [知乎-基于Simase_LSTM的计算中文句子相似度经验总结与分享](https://zhuanlan.zhihu.com/p/26996025)
2. [知乎-如何判断两段文本说的是「同一件事情」？](https://www.zhihu.com/question/56751077)
3. [Papers of Text Matching (I)](https://zhuanlan.zhihu.com/p/27441587)
4. [Papers of Text Matching (II)](https://zhuanlan.zhihu.com/p/27443681)
5. [《Bilateral Multi-Perspective Matching for Natural Language Sentences》读书笔记](https://zhuanlan.zhihu.com/p/26548034)





## Run the model
### Train
0. `mkdir log & mkdir checkpoints` 创建文件夹
1. `bash process.sh <config>` 预处理数据（根据`config.py`中定义的名为`<config>`的设定来处理）
2. `python2 train.py --config <config>` 训练名为`<config>`的模型

Other options:

* `--disable_cuda`, 禁止使用GPU, 如果没有GPU会自动使用CPU, 不需要加这个
* `--cuda_num <n>`, 使用编号为n的GPU
* `--suffix <name>` 为模型命名创建后缀，默认为'default'








## Notes & Ideas
1. 在计算score的时候，给pos类的权重越大，threshold（大于这个值认为label是1）也需要越大。
2. Word-to-Mover Distance
3. Decomposible attention
4. 去除duplicates以后用RNN没有什么提升，但可以尝试用来计算其他feature





## Results
| 日期       | 名字                         | valid成绩 | test成绩 |      |
| ---------- | ---------------------------- | --------- | -------- | ---- |
| 2018-05-29 | siamese_char_best            | 0.52?     | 0.6188   |      |
| 2018-05-30 | att_siamese_default_best     | 0.54?     | 0.5808   |      |
| 2018-05-30 | att_siamese_small_embed_best | 0.54?     | 0.6142   |      |
| 2018-05-31 | siamese_large_best           |           | 0.6265   |      |
|            |                              |           |          |      |
|            |                              |           |          |      |
|            |                              |           |          |      |
|            |                              |           |          |      |
|            |                              |           |          |      |

1. 
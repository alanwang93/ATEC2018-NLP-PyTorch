#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

from .extractor import Extractor
from ..vocab import Vocab
import jieba
import re
import numpy as np
UNK_IDX = 0

class WordBoolExtractor(Extractor):

    def __init__(self):
        Extractor.__init__(self, name="WordBoolExtractor")
        # 100
        self.words1 = [u'花呗', u'借呗', u'吗', u'还', u'还款', u'蚂蚁', u'额度', u'分期', u'开通', u'是',\
                 u'没有', u'有', u'付款', u'多少', u'逾期', u'提前', u'会', u'退款', u'银行卡', u'收款',\
                 u'余额', u'自动', u'如何', u'和', u'都', u'淘宝', u'临时', u'影响', u'还是', u'就',\
                 u'利息', u'绑定', u'信用卡', u'手续费', u'信用', u'扣款', u'怎样', u'商家', u'码', u'收钱',\
                 u'恢复', u'完', u'扣', u'是不是', u'最低', u'也', u'手机', u'手机号', u'冻结', u'账',\
                 u'还有', u'只能', u'宝', u'款', u'一次', u'不用', u'多', u'设置', u'不是', u'另', \
                 u'提额', u'付', u'算', u'期', u'怎么回事', u'又', u'不会', u'为啥', u'日期', u'下个月', \
                 u'交', u'积分', u'电费', u'对', u'怎么样', u'一样', u'每个', u'限制', u'短信', u'一个月', \
                 u'商品', u'二维码', u'你', u'为何', u'就是', u'更改', u'一次性', u'两个', u'方式', u'别人', \
                 u'可用', u'有没有', u'直接', u'订单', u'提', u'卡', u'一', u'那么', u'有钱', u'先']
        # 24
        self.words2 = [u'*', u'没有', u'不', u'在', u'不了', u'支付', u'月', u'到', u'显示', u'会', \
                 u'退款', u'时候', u'关闭', u'如何', u'和', u'借', u'一个', u'淘宝', u'临时', u'信用卡', \
                 u'取消', u'支持', u'恢复', u'冻结']

        self.feat_names = ['power_words']
        self.feat_levels = ['p']
        self.feat_lens = [len(self.words1)+len(self.words2)]

    def extract(self, data):
        funcs = ['f1', 'f2']
        feats = []
        s1s = []
        s2s = []
        for ins in data:
            s1s.append(ins['s1_word'])
            s2s.append(ins['s2_word'])
        for func in funcs:
            feats.append(self.gather(func, s1s, s2s))
        feats = np.concatenate(feats, axis=1)
        return feats

    def f1(self, s1, s2):
        """ return 1. if only one of the sentence has the word """

        d = len(self.words1)
        feat = np.zeros((d))
        for i, w in enumerate(self.words1):
            w = w.encode('utf8')
            if (w in s1 and w not in s2) or (w in s2 and w not in s1):
                feat[i] = 1.
        return feat

    def f2(self, s1, s2):
        """ return 1. if only both the 2 sentences have the word """

        d = len(self.words2)
        feat = np.zeros((d))
        for i, w in enumerate(self.words2):
            w = w.encode('utf8')
            if w in s1 and w in s2:
                feat[i] = 1.
        return feat

    def gather(self, funcname, s1s, s2s):
        n = len(s1s)
        feat = []
        for s1, s2 in zip(s1s, s2s):
            feat.append(getattr(self, funcname)(s1, s2))
        return np.asarray(feat)


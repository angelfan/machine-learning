# coding: utf-8

import re
import math


def get_words(doc):
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]

    return dict([(w, 1) for w in words])


class Classifier:
    def __init__(self, get_features):
        self.fc = {}
        self.cc = {}
        self.get_features = get_features

    def train(self, item, cat):
        features = self.get_features(item)
        for f in features:
            self.incf(f, cat)
        self.incc(cat)

    # 增加对特征/分类组合的计数值
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # 增加对某一分类的计数值
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # 某一特征出现于某一分类中的次数
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    # 属于某一分类的内容项数量
    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0.0

    # 所有内容项数量
    def total_count(self):
        return sum(self.cc.values())

    # 所有分类的列表
    def categories(self):
        return self.cc.keys()


def sample_train(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('by pharmaceuticals now', 'bad')
    cl.train('make quick money at the onlune casino', 'bad')
    cl.train('the quick brown fox iumps over the lazy dog', 'good')

cl = Classifier(get_words)
sample_train(cl)
print cl.fcount('quick', 'good')
print cl.fcount('quick', 'bad')
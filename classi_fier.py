# coding: utf-8

import re


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

    # 计算单词在分类中出现的概率
    def fprob(self, f, cat):
        if self.catcount(cat) == 0: return 0
        return self.fcount(f, cat) / self.catcount(cat)

    # 加权
    def weighted_prob(self, f, cat, prf, weight=1.0, ap=0.5):
        basic_prob = prf(f, cat)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        return (weight * ap + totals * basic_prob) / (weight + totals)

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


class NavieBayes(Classifier):
    def __init__(self, get_features):
        self.fc = {}
        self.cc = {}
        self.get_features = get_features
        self.thresholds = {}

    def doc_prob(self, item, cat):
        features = self.get_features(item)
        p = 1
        for f in features: p *= self.weighted_prob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        cat_prob = self.catcount(cat) / self.total_count()
        doc_prob = self.doc_prob(item, cat)
        return cat_prob * doc_prob

    def set_thresholds(self, cat, t):
        self.thresholds[cat] = t

    def get_thresholds(self, cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    def classify(self, item, default=None):
        probs = {}
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        for cat in probs:
            if cat == best: continue
            if probs[cat] * self.get_thresholds(best) > probs[best]: return default
        return best


def sample_train(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('by pharmaceuticals now', 'bad')
    cl.train('make quick money at the onlune casino', 'bad')
    cl.train('the quick brown fox iumps over the lazy dog', 'good')


bys = NavieBayes(get_words)
sample_train(bys)
print bys.classify('quick rabbit')
print bys.classify('quick money')

# coding: utf-8

import re
import math


def get_words(doc):
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]

    return dict([(w, 1) for w in words])


class Classifier:
    def __init__(self, get_features):
        self.ap = 0.5
        self.weight = 1.0
        self.features = {}
        self.categories = {}
        self.get_features = get_features  # 提取特征的func

    # 对样本进行训练
    def train(self, item, cat):
        features = self.get_features(item)
        for f in features:
            self.inc_feat(f, cat)
        self.inc_cat(cat)

    def set_ap(self, ap, weight):
        self.ap = ap
        self.weight = weight

    # 加权后的概率
    def weighted_prob(self, feat, cat, feat_prob):
        basic_prob = feat_prob(feat, cat)
        totals = sum([self.feat_count(feat, c) for c in self.cats()])
        return (self.weight * self.ap + totals * basic_prob) / (self.weight + totals)

    # 增加对特征/分类组合的计数值
    def inc_feat(self, feat, cat):
        self.features.setdefault(feat, {})
        self.features[feat].setdefault(cat, 0)
        self.features[feat][cat] += 1

    # 增加对某一分类的计数值
    def inc_cat(self, cat):
        self.categories.setdefault(cat, 0)
        self.categories[cat] += 1

    # 某一特征出现于某一分类中的次数
    def feat_count(self, feat, cat):
        if feat in self.features and cat in self.features[feat]:
            return float(self.features[feat][cat])
        return 0.0

    # 计算特征在分类中出现的概率
    def feat_prob(self, feat, cat):
        if self.cat_count(cat) == 0: return 0
        return self.feat_count(feat, cat) / self.cat_count(cat)

    # 属于某一分类的内容项数量
    def cat_count(self, cat):
        if cat in self.categories:
            return float(self.categories[cat])
        return 0.0

    # 所有内容项数量
    def total_count(self):
        return sum(self.categories.values())

    # 所有分类的列表
    def cats(self):
        return self.categories.keys()


class NavieBayes(Classifier):
    def __init__(self, get_features):
        Classifier.__init__(self, get_features)
        self.thresholds = {}

    # 对Item进行分类
    def classify(self, item, default=None):
        probs = {}
        max_prob = 0.0

        # 寻找概率最大的分类
        for cat in self.cats():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max_prob:
                max_prob = probs[cat]
                best = cat

        # 确保概率值超出阈值*次大概率值
        for cat in probs:
            if cat == best: continue
            if probs[cat] * self.get_thresholds(best) > probs[best]: return default
        return best

    # 设定阈值
    def set_thresholds(self, cat, t):
        self.thresholds[cat] = t

    # Item属于该分类的概率
    # P(Category | Item)
    def prob(self, item, cat):
        item_prob = self.item_prob(item, cat)
        cat_prob = self.cat_count(cat) / self.total_count()
        return item_prob * cat_prob

    # P(Item | Category)
    def item_prob(self, item, cat):
        features = self.get_features(item)
        p = 1
        for feat in features: p *= self.weighted_prob(feat, cat, self.feat_prob)
        return p

    def get_thresholds(self, cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]


class Fisher(Classifier):
    def __init__(self, get_features):
        Classifier.__init__(self, get_features)
        self.minimums = {}

    def classify(self, item, default=None):
        best = default
        max = 0.0
        for c in self.cats():
            p = self.prob(item, c)
            if p > self.get_min(c) and p > max:
                best = c
                max = p

        return best

    def prob(self, item, cat):
        p = 1
        features = self.get_features(item)
        for feat in features:
            p *= self.weighted_prob(feat, cat, self.cat_prob)

        feat_score = -2 * math.log(p)
        return self.invchi2(feat_score, len(features) * 2)

    def invchi2(self, chi, df):
        m = chi / 2.0
        sum = term = math.exp(-m)
        for i in range(1, df / 2):
            term *= m / i
            sum += term
        return min(sum, 1.0)

    def cat_prob(self, feat, cat):
        # 特征在该分类中出现的概率
        clf = self.feat_prob(feat, cat)
        if clf == 0: return 0

        # 特征在所有分类中出现的概率
        freq_sum = sum([self.feat_prob(feat, c) for c in self.cats()])

        # 分类中出现频率 除以 总频率
        return clf / freq_sum

    def set_min(self, cat, min):
        self.minimums[cat] = min

    def get_min(self, cat):
        if cat not in self.minimums: return 0
        return self.minimums[cat]


"""
高斯模型:
当特征是连续变量的时候, 所得到的条件概率也难以描述真实情况。
所以处理连续的特征变量，应该采用高斯模型。
"""
import numpy as np


def get_features(feats):
    return feats


class Gaussian(Classifier):
    def __init__(self, get_features):
        Classifier.__init__(self, get_features)

    def classify(self, item):
        probs = {}
        max_prob = 0.0

        for cat in self.cats():
            probs[cat] = self.item_prob(item, cat)
            if probs[cat] > max_prob:
                max_prob = probs[cat]
                best = cat

        return best

    def inc_feat(self, feat, cat, values):
        self.features.setdefault(feat, {})
        self.features[feat].setdefault(cat, [])
        self.features[feat][cat].append(values)

    # 对样本进行训练
    def train(self, item, cat):
        features = self.get_features(item)
        for key in features:
            self.inc_feat(key, cat, features[key])
        self.inc_cat(cat)

    def item_prob(self, item, cat):
        features = self.get_features(item)
        p = 1
        for feat in features:
            feat_prob = self.feat_prob(feat, cat)
            p *= self.gaussian(feat_prob[0], feat_prob[1], features[feat])
        return p

    def feat_prob(self, feat, cat):
        feats = self.features[feat][cat]
        mu = np.mean(feats)
        sigma = np.std(feats)
        return mu, sigma

    def gaussian(self, mu, sigma, value):
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (value - mu) ** 2 / (2 * sigma ** 2))


def sample_train(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('by pharmaceuticals now', 'bad')
    cl.train('make quick money at the onlune casino', 'bad')
    cl.train('the quick brown fox iumps over the lazy dog', 'good')


def gaussian_train(gass):
    gass.train({'height': 180, 'weight': 70}, 'M')
    gass.train({'height': 179, 'weight': 69}, 'M')
    gass.train({'height': 178, 'weight': 68}, 'M')
    gass.train({'height': 179, 'weight': 80}, 'M')
    gass.train({'height': 170, 'weight': 60}, 'M')

    gass.train({'height': 155, 'weight': 45}, 'F')
    gass.train({'height': 165, 'weight': 66}, 'F')
    gass.train({'height': 158, 'weight': 48}, 'F')
    gass.train({'height': 150, 'weight': 60}, 'F')


bys = NavieBayes(get_words)
sample_train(bys)
print bys.classify('quick rabbit')
print bys.classify('quick money')
print bys.classify('pharmaceuticals now')

fisher = Fisher(get_words)
sample_train(fisher)
print fisher.classify('quick money')
fisher.set_min('bad', 0.8)
print fisher.classify('quick money')

gaussian = Gaussian(get_features)
gaussian_train(gaussian)
print gaussian.classify({'height': 155, 'weight': 45})  # 女 标准 体型
print gaussian.classify({'height': 175, 'weight': 45})  # 女 高 瘦
print gaussian.classify({'height': 168, 'weight': 80})  # 男 小胖子
print gaussian.classify({'height': 190, 'weight': 60})  # 男 虽然我长的高 但是我瘦啊

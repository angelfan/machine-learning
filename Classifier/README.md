从训练数据中我们可以知道的条件
该特征属于该类别的概率
```该特征在分类别中出现的次数 / 分类的总数```
我们称之为条件概率
通常记作`P(A|B)` 即 `P(word | classification)`
意思是说假如
```python
[
    ['good work', 'happy'],
    ['good luck', 'happy'],
    ['good morning', 'happy'],
    ['it is fine', 'happy'],
    ['can not', 'sad'],
]
P('good' | 'happy') = 3 / 4
```

为了解决对极少出现的单词变得异常敏感(假设某个单词只在一个分类中出现了一次)
可以对其进行加权
```
# assume_prob 假设概率 推荐的初始值 0.5
# weight 权重 权重为1表示假设概率的权重与一个单词相当
# prob 上文的 P(word | classification)
# totals 单词在所有分类中出现的次数
(weight * assume_prob + totals * prob) / ( weight + totals )

这样当某个单词只在某一分类中出现一次的时候
比如
['good work', 'happy']
['bad work', 'sad']

P(good | happy) 加权后的概率 (1 * 0.5 + 1 * 1) / (1 + 1) = 0.75
P(good | sad)   加权后的概率 (1 * 0.5 + 1 * 0) / (1 + 1) = 0.25
```


朴素贝叶斯对条件概率分布作出了独立性的假设, 即某个特征发生的概率对其它特征发生的概率没有影响
所以判断一个特征组属于哪个类别其实就是分别计算各个特征在某一分类下的概率, 然后简单相乘
最终选择发生概率最大的分类作为输出结果
P(document | category) 叫做事件category发生下事件document的条件概率
```python
for f in features: p *= P(f | happy)
```

P(category | document) = P(document | category) * P(category) / P(document)
P(document) 是没必要计算的项目
P(category) = 类别项目中 / 总项目数
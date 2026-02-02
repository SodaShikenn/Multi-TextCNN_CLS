# Model Documentation

## 1. Question Classification Model

### 1.1 Classification Labels

```
0 - Efficacy (功效)
1 - Target Users (适⽤⼈群)
2 - Usage Method (使用方法)
3 - Others (其他)
4 - Attributes (属性)
5 - User Experience (使用感受)
6 - Adverse Reactions (不良反应)
7 - Competitor Comparison (竞品对比)
8 - Packaging (包装)
9 - Price (价格)
10 - Sales Channel (渠道)
11 - Logistics (物流)
```

### 1.2 Usage Example

```python
from models.classification.functions import predict

label_ids, label_texts = predict([
    '这款面霜小孩子可以用吗？会不会有副作用？',
    '我怎么闻着是酒精的味道？正常吗？',
    '请问，春天可以用嘛?',
    '这款油腻吗？',
    '保湿效果怎么样？'
])

print(label_ids)   # Classification IDs: [[1, 6], [4], [1], [5], [0]]
print(label_texts) # Classification names: [['适⽤⼈群', '不良反应'], ['属性'], ['适⽤⼈群'], ['使用感受'], ['功效']]
```

## 2. Sentiment Analysis Model

### 2.1 Sentiment Labels

```
0 - Neutral (中评)
1 - Positive (好评)
2 - Negative (差评)
```

### 2.2 Usage Example

```python
from models.sentiment.functions import predict

label_ids, label_texts = predict([
    ['是正品吗？', '说不好，反正包装很糙'],
    ['是正品吗？', '太辣鸡了。'],
    ['亲们祛斑效果怎么样？', '效果不错。'],
    ['这个和john jeff哪个好用', '感觉差不多'],
])

print(label_ids)   # [0, 2, 1, 0]
print(label_texts) # ['中评', '差评', '好评', '中评']
```

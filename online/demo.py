from models.classification.functions import predict

label_ids, label_texts = predict([
    '这款面霜小孩子可以用吗？会不会有副作用？',
    '我怎么闻着是酒精的味道？正常吗？',
    '请问，春天可以用嘛?',
    '这款油腻吗？',
    '保湿效果怎么样？'
])


# from models.sentiment.functions import predict

# label_ids, label_texts = predict([
#     ['是正品吗？', '说不好，反正包装很糙'],
#     ['是正品吗？', '太辣鸡了。'],
#     ['亲们祛斑效果怎么样？', '效果不错'],
#     ['这个和john jeff哪个好用', '感觉差不多'],
# ])

print(label_ids)
print(label_texts)


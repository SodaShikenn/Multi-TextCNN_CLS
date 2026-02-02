from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_99.pth', map_location=DEVICE))
    # model = torch.load(MODEL_DIR + '20.pth', map_location=DEVICE)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    texts = [
        # Multi-label examples: 适用人群 + 不良反应
        '这款面霜小孩子可以用吗？会不会有副作用？',
        # 属性 (酒精味道)
        '我怎么闻着是酒精的味道？正常吗？',
        # 使用方法
        '请问，春天可以用嘛?',
        # Multi-label: 功效 + 不良反应
        '祛痘效果好吗？用了会不会过敏？',
        # Multi-label: 价格 + 渠道
        '这个多少钱？在哪里可以买到正品？',
        # Multi-label: 包装 + 物流
        '包装怎么样？发货快吗？',
        # Multi-label: 功效 + 竞品对比
        '美白效果好吗？和SK2比哪个好？',
        # Multi-label: 适用人群 + 使用方法
        '孕妇能用吗？一天涂几次？',
        # Multi-label: 使用感受 + 属性
        '用起来油腻吗？是什么质地的？',
        # Multi-label: 功效 + 适用人群 + 不良反应
        '敏感肌可以用吗？能祛斑吗？会刺激皮肤吗？',
    ]

    batch_input_ids = []
    batch_mask = []
    for text in texts:
        tokened = tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        batch_input_ids.append(input_ids[:TEXT_LEN])
        batch_mask.append(mask[:TEXT_LEN])

    batch_input_ids = torch.tensor(batch_input_ids)
    batch_mask = torch.tensor(batch_mask)

    batch_input_ids = batch_input_ids.to(DEVICE)
    batch_mask = batch_mask.to(DEVICE)

    pred = model(batch_input_ids, batch_mask)

    label_ids = []
    label_names = []

    for row in pred:
        item = np.where(row.cpu().data.numpy() > CLS_PROB_BAR, 1, 0)
        lids = seq2ids(item)
        label_ids.append(lids)
        label_names.append([id2label[l] for l in lids])
    print(label_ids, label_names)
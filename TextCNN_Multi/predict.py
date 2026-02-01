from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_20.pth', map_location=DEVICE))
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    texts = [
        '可以祛斑吗？',
        '小孩能不能用？',
        '可以祛痘吗，有没有副作用？',
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

    batch_input_ids = torch.tensor(batch_input_ids).to(DEVICE)
    batch_mask = torch.tensor(batch_mask).to(DEVICE)

    with torch.no_grad():
        pred = model(batch_input_ids, batch_mask)
        pred_probs = torch.sigmoid(pred)  # Multi-label: use sigmoid
        pred_labels = (pred_probs > CLS_PROB_BAR).int()  # Threshold

    # Print predictions for each text
    for i, text in enumerate(texts):
        labels = [id2label[j] for j in range(NUM_CLASSES) if pred_labels[i][j] == 1]
        print(f"{text} -> {labels}")
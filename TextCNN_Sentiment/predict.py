from config import *
from utils import *
from model import *

if __name__ == '__main__':

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_39.pth', map_location=DEVICE))
    # model = torch.load(MODEL_DIR + '20.pth', map_location=DEVICE)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    texts = [
        ['是正品吗？', '说不好，反正包装很糙'],
        ['是正品吗？', '太辣鸡了。'],
        ['亲们祛斑效果怎么样？', '效果不错'],
        ['这个和john jeff哪个好用', '感觉差不多'],
    ]

    batch_input_ids = []
    batch_mask = []
    for question_content, answer_content in texts:
        # 问题文本编码
        question_tokened = tokenizer(question_content)
        question_input_ids = question_tokened['input_ids']
        question_mask = question_tokened['attention_mask']
        if len(question_input_ids) < QUESTION_TEXT_LEN:
            pad_len = (QUESTION_TEXT_LEN - len(question_input_ids))
            question_input_ids += [BERT_PAD_ID] * pad_len
            question_mask += [0] * pad_len
        else:
            question_input_ids = question_input_ids[:QUESTION_TEXT_LEN]
            question_mask = question_mask[:QUESTION_TEXT_LEN]
        # 答案文本编码
        answer_tokened = tokenizer(answer_content)
        answer_input_ids = answer_tokened['input_ids']
        answer_mask = answer_tokened['attention_mask']
        if len(answer_input_ids) < QUESTION_TEXT_LEN:
            pad_len = (QUESTION_TEXT_LEN - len(answer_input_ids))
            answer_input_ids += [BERT_PAD_ID] * pad_len
            answer_mask += [0] * pad_len
        else:
            answer_input_ids = answer_input_ids[:QUESTION_TEXT_LEN]
            answer_mask = answer_mask[:QUESTION_TEXT_LEN]
        # 问题+答案，作为整体传给模型
        input_ids = question_input_ids + answer_input_ids
        mask = question_mask + answer_mask
        # 压入batch，批量操作
        batch_input_ids.append(input_ids)
        batch_mask.append(mask)

    batch_input_ids = torch.tensor(batch_input_ids)
    batch_mask = torch.tensor(batch_mask)

    batch_input_ids = batch_input_ids.to(DEVICE)
    batch_mask = batch_mask.to(DEVICE)

    pred = model(batch_input_ids, batch_mask)
    pred_ = torch.argmax(pred, dim=1)

    print(pred_, [ID2LABEL[l] for l in pred_])
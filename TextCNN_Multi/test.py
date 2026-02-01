from config import *
from utils import *
from model import *

if __name__ == '__main__':

    id2label, _ = get_label()

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = TextCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR + 'model_weights_20.pth', map_location=DEVICE))

    y_pred = []
    y_true = []

    with torch.no_grad():
        for b, (input, mask, target) in enumerate(test_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            test_pred = model(input, mask)
            loss = model.loss_fn(test_pred, target)

            print('>> batch:', b, 'loss:', round(loss.item(), 5))

            test_pred_ = torch.sigmoid(test_pred).cpu().data.numpy()  # Multi-label

            y_pred.append(test_pred_)
            y_true.append(target.cpu().data.numpy())

    import numpy as np
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    report = evaluate(y_pred, y_true)
    print(f"Precision: {report['precision']:.4f}, Recall: {report['recall']:.4f}, F1: {report['f1_score']:.4f}")
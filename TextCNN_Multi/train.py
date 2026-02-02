from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Used as 
    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # loss_fn = nn.CrossEntropyLoss()

    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = model.loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 50 != 0:
                continue

            report = evaluate(pred.cpu().data.numpy(), target.cpu().data.numpy())

            print(
                '>> epoch:', e,
                'batch:', b,
                'loss:', round(loss.item(), 5),
                'train_f1:', report['f1_score'],
            )

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

                y_pred += test_pred.data.tolist()
                y_true += target.data.tolist()

        report = evaluate(y_pred, y_true)
        print('test_f1:', report['f1_score'])

        # torch.save(model, MODEL_DIR + f'{e}.pth')
        # 保存模型参数
        torch.save(model.state_dict(), MODEL_DIR + f'model_weights_multi_{e}.pth')
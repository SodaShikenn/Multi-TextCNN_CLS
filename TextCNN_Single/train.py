from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()  # Single-label classification

    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)  # CrossEntropyLoss expects long targets

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 50 != 0:
                continue

            y_pred = torch.argmax(pred, dim=1)  # Single-label: use argmax
            report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), output_dict=True)

            with torch.no_grad():
                test_input, test_mask, test_target = next(iter(test_loader))
                test_input = test_input.to(DEVICE)
                test_mask = test_mask.to(DEVICE)
                test_target = test_target.to(DEVICE)
                test_pred = model(test_input, test_mask)
                test_pred_ = torch.argmax(test_pred, dim=1)  # Single-label: use argmax
                test_report = evaluate(test_pred_.cpu().data.numpy(), test_target.cpu().data.numpy(), output_dict=True)

            print(
                '>> epoch:', e,
                'batch:', b,
                'loss:', round(loss.item(), 5),
                'train_acc:', report['accuracy'],
                'test_acc:', test_report['accuracy'],
            )
        # 保存模型
        # torch.save(model, MODEL_DIR + f'{e}.pth')
        # 保存模型参数
        torch.save(model.state_dict(), MODEL_DIR + f'model_weights_{e}.pth')
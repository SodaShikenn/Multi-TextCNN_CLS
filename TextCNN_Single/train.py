from config import *
from utils import *
from model import *

if __name__ == '__main__':
    id2label, _ = get_label()

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Used as dev
    dev_dataset = Dataset('test')
    dev_loader = data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()  # Single-label classification

    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 50 != 0:
                continue

            y_pred = torch.argmax(pred, dim=1)
            report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), output_dict=True)

            print(
                '>> epoch:', e,
                'batch:', b,
                'loss:', round(loss.item(), 5),
                'train_acc:', round(report['accuracy'], 4),
            )

        # Dev evaluation at end of each epoch
        y_pred = []
        y_true = []

        with torch.no_grad():
            for b, (input, mask, target) in enumerate(dev_loader):
                input = input.to(DEVICE)
                mask = mask.to(DEVICE)
                target = target.to(DEVICE)

                dev_pred = model(input, mask)
                dev_pred_ = torch.argmax(dev_pred, dim=1)

                y_pred += dev_pred_.cpu().data.tolist()
                y_true += target.cpu().data.tolist()

        report = evaluate(y_pred, y_true, output_dict=True)
        print('>> epoch:', e, 'dev_acc:', round(report['accuracy'], 4))
        print(evaluate(y_pred, y_true, target_names=id2label))

        # Save model weights
        torch.save(model.state_dict(), MODEL_DIR + f'model_weights_single_{e}.pth')

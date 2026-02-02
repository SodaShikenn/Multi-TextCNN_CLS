from config import *
from utils import *
from model import *

if __name__ == '__main__':

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load test set (used as validation set)
    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

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
                'train_acc:', report['accuracy'],
            )

        # Model validation
        y_pred = []
        y_true = []

        with torch.no_grad():
            for b, (input, mask, target) in enumerate(test_loader):
                input = input.to(DEVICE)
                mask = mask.to(DEVICE)
                target = target.to(DEVICE)

                test_pred = model(input, mask)
                loss = loss_fn(test_pred, target)

                # print('>> batch:', b, 'loss:', round(loss.item(), 5))

                test_pred_ = torch.argmax(test_pred, dim=1)

                y_pred += test_pred_.data.tolist()
                y_true += target.data.tolist()

        report = evaluate(y_pred, y_true, ID2LABEL, output_dict=True)
        print('test_acc:', report['accuracy'])

        # torch.save(model, MODEL_DIR + f'{e}.pth')
        # Save model weights
        torch.save(model.state_dict(), MODEL_DIR + f'model_weights_{e}.pth')
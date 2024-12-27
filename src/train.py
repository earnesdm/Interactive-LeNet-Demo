import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.datasets as datasets

import matplotlib.pyplot as plt

from model import LeNet


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def train(model, optimizer, loader_train, loader_val, epochs=1, print_every=100):
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    val_accs = []
    for e in range(epochs):
        print('-' * 128)
        print('Epoch {}/{}'.format(e, epochs))
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = loss_fn(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()
        val_accs.append(check_accuracy(loader_val, model))
    return val_accs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    my_transform = T.Compose([
                    T.ToTensor(),
                    T.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-5, 5, -5, 5)),
                    T.RandomPerspective(),
                    #T.RandomErasing()
                ])

    mnist_train = datasets.MNIST('.venv', download=True, train=True, transform=my_transform)
    loader_train = DataLoader(mnist_train, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(50000)))

    mnist_val = datasets.MNIST('.venv', download=True, train=True, transform=my_transform)
    loader_val = DataLoader(mnist_val, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(range(50000, 60000)))

    #batch = next(iter(loader_train))
    #print(batch[0].shape)
    #plt.imshow(batch[0][1].reshape(1, 28, 28).permute(1, 2, 0).numpy(), cmap="grey")
    #plt.savefig(f"test.png")

    model = LeNet()

    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, loader_train, loader_val, epochs=15, print_every=200)

    torch.save(model.state_dict(), "LeNet_Params5.pt")

import torch
from torch.autograd import Variable


def evaluate(model, loader):
    avg_accuracy = 0
    for data in loader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        labels = labels.view(-1)

        predictions = model(images)
        _, predictions = predictions.max(dim=1)
        correct = (predictions == labels).float()
        accuracy = correct.mean().data[0]

        avg_accuracy += accuracy * images.size(0) / len(loader.dataset)

    return avg_accuracy

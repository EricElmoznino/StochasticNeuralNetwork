import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Model
from dataset import MNSTDataset
from evaluater import evaluate

epochs = 1
batch_size = 32

model_deterministic = Model(False)
if torch.cuda.is_available():
    model_deterministic.cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_deterministic.parameters())

train_set = MNSTDataset('data/train')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = MNSTDataset('data/test')
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)

for epoch in range(1, epochs + 1):
    print('Starting Epoch %d' % epoch)
    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        labels = labels.view(-1)

        predictions = model_deterministic(images)
        loss = loss_func(predictions, labels)
        model_deterministic.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            loss = loss.data[0]
            print('[%d / %d] Loss: %f' %
                  ((epoch - 1) * len(train_loader) + i, epochs * len(train_loader), loss))

    print('\nFinished Epoch %d' % epoch)
    acc = evaluate(model_deterministic, test_loader)
    print('Test Accuracy: %.2f\n' % (acc * 100))

model_deterministic.eval()
state = model_deterministic.state_dict()
torch.save(state, 'saved_model.pt')

model_stochastic = Model(True)
if torch.cuda.is_available():
    model_stochastic.cuda()
model_stochastic.eval()
model_stochastic.load_state_dict(state)

print('Stochastic Model Evaluation')
acc = evaluate(model_stochastic, test_loader)
print('Test Accuracy: %.2f\n' % (acc * 100))

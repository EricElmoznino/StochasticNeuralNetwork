import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tr
from PIL import Image
import os


class MNSTDataset(Dataset):

    def __init__(self, dir):
        self.data = os.listdir(dir)
        self.data = [d for d in self.data if '.jpg' in d]
        self.data = [{'image': os.path.join(dir, d),
                      'label': int(d.split('_')[0])}
                     for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]['image']
        image = Image.open(image).convert('L')
        image = tr.to_tensor(image)

        label = self.data[item]['label']
        label = torch.LongTensor([int(label)])

        return image, label

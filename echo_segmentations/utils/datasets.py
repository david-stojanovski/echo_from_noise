import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DiffusionDataset(Dataset):

    def __init__(self, image_dir, img_transform=None, label_transform=None, device=torch.device('cpu')):
        self.image_dir = image_dir
        self.image_path = os.listdir(self.image_dir % "images")
        self.label_path = os.listdir(self.image_dir % "annotations")
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.device = device

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_fp = os.path.join(self.image_dir % 'images', self.image_path[index])

        out_img = Image.open(image_fp).convert('L')
        out_img = out_img.resize((256, 256), Image.NEAREST)
        out_img = np.array(out_img)
        out_img = np.expand_dims(out_img, axis=-1)
        # transpose out_img preserving dimensions
        label_fp = os.path.join(self.image_dir % 'annotations', self.image_path[index])
        label = Image.open(label_fp).convert('L')
        label = label.resize((256, 256), Image.NEAREST)
        label = np.array(label)
        label = np.expand_dims(label, axis=-1)

        out_img = torch.tensor(out_img).permute(2, 0, 1)
        label = torch.tensor(label).permute(2, 0, 1)
        seed = np.random.randint(2147483647)  # set rand seed so that the image and label are transformed the same way
        random.seed(seed)
        torch.manual_seed(seed)
        if self.img_transform:
            out_img = self.img_transform(out_img)

        random.seed(seed)
        torch.manual_seed(seed)
        if self.label_transform:
            label = self.label_transform(label)

        return out_img, label

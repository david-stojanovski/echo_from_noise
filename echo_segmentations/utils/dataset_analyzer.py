import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils.data_transforms as data_transforms
from utils.datasets import DiffusionDataset


def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
        snd_moment - fst_moment ** 2)
    return mean, std


def get_mean_and_std(train_dir, val_dir, device, batch_size, num_workers):
    train_val_img_transforms = data_transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    train_val_label_transforms = data_transforms.Compose([
        data_transforms.TensorSqueeze()
    ])

    train_dataset = DiffusionDataset(train_dir, img_transform=train_val_img_transforms,
                                     label_transform=train_val_label_transforms,
                                     device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_workers)

    val_dataset = DiffusionDataset(val_dir, img_transform=train_val_img_transforms,
                                   label_transform=train_val_label_transforms,
                                   device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=num_workers)

    mean_train, std_train = batch_mean_and_sd(train_loader)
    mean_val, std_val = batch_mean_and_sd(val_loader)
    mean_total = (mean_train + mean_val) / 2.
    std_total = (std_train + std_val) / 2.
    print("mean and std: \n", mean_total, std_total)
    return mean_total[0], std_total[0]

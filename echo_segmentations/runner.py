import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from monai.networks.nets import UNet
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import utils.data_transforms as data_transforms
import utils.dataset_analyzer as dataset_analyzer
from utils.datasets import DiffusionDataset
from utils.epoch_cycle import run_cycle
from utils.network_utils import add_data2tb


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser for Echo segmentation')
    parser.add_argument('--data-dir',
                        dest='data_dir',
                        help='path to data directory',
                        default=r'/home/ds17/Documents/phd/echo_from_noise/data_preparation/seg_data_trial/2CH_ES/%s/',
                        type=str)
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        help='path to save directory',
                        default=os.path.join(os.getcwd(), 'output'),
                        type=str)
    parser.add_argument('--model-path',
                        dest='load_model_path',
                        help='path to pretrained model',
                        default=None)
    parser.add_argument('--num-classes',
                        dest='num_classes',
                        help='number of label classes',
                        default=4,
                        type=int)
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default="cuda:0",
                        type=str)
    parser.add_argument('--compile-model',
                        dest='compile_model',
                        help='Whether to perform pytorch 2.0 automatic model compilation',
                        default=True,
                        type=bool)
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        default=1,
                        type=int)
    parser.add_argument('--num-workers',
                        dest='num_workers',
                        help='number of cpu cores to use',
                        default=10,
                        type=int)
    parser.add_argument('--epochs',
                        dest='num_epochs',
                        help='number of epochs to run training for',
                        default=300,
                        type=int)
    parser.add_argument('--lr',
                        dest='lr',
                        help='learning rate',
                        default=1e-3,
                        type=float)
    parser.add_argument('--min-lr',
                        dest='min_lr',
                        help='final learning rate',
                        default=1e-4,
                        type=float)
    parser.add_argument('--test-viz-num',
                        dest='test_batch_size',
                        help='Number of example inference cases to plot at end of training',
                        default=8,
                        type=int)
    parser.add_argument('--pin-memory',
                        dest='pin_memory',
                        help='Whether to pin memory of dataloaders',
                        default=True,
                        type=bool)
    parser.add_argument('--train-shuffle',
                        dest='train_shuffle',
                        help='Whether to pin shuffle training/validation data',
                        default=True,
                        type=bool)
    parser.add_argument('--test-shuffle',
                        dest='test_shuffle',
                        help='Whether to pin shuffle testing data',
                        default=False,
                        type=bool)

    args = parser.parse_args()
    return args


def main(args):
    train_dir = os.path.join(args.data_dir, "training")
    val_dir = os.path.join(args.data_dir, "validation")
    test_dir = os.path.join(args.data_dir, "testing")

    device = args.gpu_id if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print('Running dataset analyzer')
    dataset_mean, dataset_std = dataset_analyzer.get_mean_and_std(train_dir, val_dir, device, batch_size=args.batch_size,
                                                 num_workers=args.num_workers)

    train_val_img_transforms = data_transforms.Compose([
        transforms.ToPILImage(),
        data_transforms.GammaAdjust((0.5, 3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.NEAREST),  # in degrees
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2),
                                interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])
    train_val_label_transforms = data_transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.NEAREST),  # in degrees
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2),
                                interpolation=transforms.InterpolationMode.NEAREST),
        data_transforms.TensorSqueeze()
    ])

    test_img_transforms = data_transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])
    test_label_transforms = data_transforms.Compose([
        data_transforms.TensorSqueeze()
    ])

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=args.num_classes,
        channels=(2, 4, 8, 16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2, 2, 2, 2),
        num_res_units=0,
        act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}),
    ).to(device)
    if args.compile_model:
        model = torch.compile(model)
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
        print('Model loaded from {}'.format(args.load_model_path))

    logs_path = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_path, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(logs_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logs_path, 'val'))
    test_writer = SummaryWriter(os.path.join(logs_path, 'test'))

    train_dataset = DiffusionDataset(train_dir, img_transform=train_val_img_transforms,
                                     label_transform=train_val_label_transforms,
                                     device=device)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.train_shuffle,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers)

    val_dataset = DiffusionDataset(val_dir, img_transform=train_val_img_transforms,
                                   label_transform=train_val_label_transforms,
                                   device=device)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_shuffle,
                                 pin_memory=args.pin_memory,
                                 num_workers=args.num_workers)

    test_dataset = DiffusionDataset(test_dir, img_transform=test_img_transforms, label_transform=test_label_transforms,
                                    device=device)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.test_shuffle,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=args.num_epochs,
                                  eta_min=args.min_lr)

    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('dice_mean', []),
        ('val_loss', []),
        ('val_dice_mean', []),
        ('test_loss', []),
        ('test_dice_mean', []),
    ])

    best_dice = 0
    best_epoch = 0

    for epoch in range(args.num_epochs + 1):
        train_log = run_cycle(train_data_loader, args.num_classes, device, model, criterion, optimizer, epoch,
                                args.num_epochs,
                                desc='train', lr_scheduler=scheduler)
        val_log = run_cycle(val_data_loader, args.num_classes, device, model, criterion, optimizer, epoch,
                              args.num_epochs,
                              desc='validation', lr_scheduler=scheduler)
        test_log = run_cycle(test_data_loader, args.num_classes, device, model, criterion, optimizer, epoch,
                               args.num_epochs,
                               desc='test', lr_scheduler=scheduler)

        add_data2tb(train_writer, train_log, epoch)
        add_data2tb(val_writer, val_log, epoch)
        add_data2tb(test_writer, test_log, epoch)

        print(
            '\nEpoch %d: [Train Loss %.4f] [Train Dice: %.4f] [Val Loss: %.4f] [Val Dice: %.4f] [Test Loss: %.4f] [Test Dice %.4f]'
            % (
                epoch, train_log['loss'], train_log['dice_mean'], val_log['loss'], val_log['dice_mean'],
                test_log['loss'],
                test_log['dice_mean']))
        print(
            'Epoch %d Test Dice: [Lv endo %.4f] [Lv Epi %.4f] [Lv Atrium %.4f] '
            % (epoch,
               test_log['dice_lv_endo'],
               test_log['dice_lv_epi'],
               test_log['dice_atrium']))

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['dice_mean'].append(train_log['dice_mean'])
        log['val_loss'].append(val_log['loss'])
        log['val_dice_mean'].append(val_log['dice_mean'])
        log['test_loss'].append(test_log['loss'])
        log['test_dice_mean'].append(test_log['dice_mean'])

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        pd.DataFrame(log).to_csv(os.path.join(args.output_dir, 'log.csv'), index=False)

        if val_log['dice_mean'] > best_dice:
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
            best_dice = val_log['dice_mean']
            print(
                "======================================================= \
                saved best model ========================================================")

    print('---------------------- Best epoch: %d ----------------------' % best_epoch)
    train_writer.close()
    val_writer.close()
    test_writer.close()


if __name__ == '__main__':
    # sys.argv = sys.argv + ['data-dir', '--output-dir', os.path.join(os.getcwd(), 'output_2CH_ES_test')]
    input_args = get_args_from_command_line()
    main(input_args)

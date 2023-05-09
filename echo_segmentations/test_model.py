import os
import sys
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from monai.networks.nets import UNet
from torch.utils.data import DataLoader

import utils.data_transforms as data_transforms
from utils.datasets import DiffusionDataset
from utils.epoch_cycle import run_cycle


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser for Echo segmentation')
    parser.add_argument('--data-dir',
                        dest='data_dir',
                        help='path to data directory',
                        default=r'/path/to/view_folder/%s/',
                        type=str)
    parser.add_argument('--output-dir',
                        dest='output_dir',
                        help='path to save directory',
                        default=os.path.join(os.getcwd(), 'output'),
                        type=str)
    parser.add_argument('--model-path',
                        dest='load_model_path',
                        help='path to pretrained model',
                        required=True)
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
                        default=False,
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
    parser.add_argument('--test-shuffle',
                        dest='test_shuffle',
                        help='Whether to pin shuffle testing data',
                        default=False,
                        type=bool)

    args = parser.parse_args()
    return args


def main(args):
    test_dir = os.path.join(args.data_dir, "testing")

    device = args.gpu_id if torch.cuda.is_available() else "cpu"

    num_classes = 4

    test_img_transforms = data_transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),

    ])
    test_label_transforms = data_transforms.Compose([
        data_transforms.TensorSqueeze()
    ])

    test_dataset = DiffusionDataset(test_dir, img_transform=test_img_transforms, label_transform=test_label_transforms,
                                    device=device)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.test_shuffle,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers)

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(2, 4, 8, 16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2, 2, 2, 2),
        num_res_units=0,
        act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}),
    ).to(device)

    if args.compile_model:
        model = torch.compile(model)

    state_dict = torch.load(args.load_model_path)
    # remove_prefix = '_orig_mod.'
    # state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()
    test_log = run_cycle(test_data_loader, num_classes, device, model, criterion=criterion, current_epoch=1,
                         desc='Test')

    print(
        'Test Dice: [Mean: %.4f] [Lv endo %.4f] [Lv Epi %.4f] [Lv Atrium %.4f] [Lv endo std %.4f] [Lv Epi std %.4f] [Lv Atrium std %.4f] '
        % (test_log['dice_mean'],
           test_log['dice_lv_endo'],
           test_log['dice_lv_epi'],
           test_log['dice_atrium'],
           test_log['dice_std_lv_endo'],
           test_log['dice_std_lv_epi'],
           test_log['dice_std_atrium']))


if __name__ == '__main__':
    # sys.argv = sys.argv + ['--model-path', '/path/to/model.pth']
    input_args = get_args_from_command_line()
    main(input_args)

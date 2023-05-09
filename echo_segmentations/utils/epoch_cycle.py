from collections import OrderedDict

import numpy as np
import torch
from torchmetrics.functional import dice
from tqdm import tqdm

from utils.network_utils import AverageMeter


def calc_dice_metrics_heart(individual_labels,
                            std_individual_labels,
                            b_loss,
                            dice_mean):
    avg_meters = {'loss': AverageMeter(),
                  'dice_lv_endo': AverageMeter(),
                  'dice_std_lv_endo': AverageMeter(),
                  'dice_atrium': AverageMeter(),
                  'dice_std_atrium': AverageMeter(),
                  'dice_lv_epi': AverageMeter(),
                  'dice_std_lv_epi': AverageMeter(),
                  'dice_mean': AverageMeter(),
                  }

    avg_meters['loss'].update(b_loss.item())
    avg_meters['dice_lv_endo'].update(individual_labels[1])
    avg_meters['dice_std_lv_endo'].update(std_individual_labels[1])

    avg_meters['dice_atrium'].update(individual_labels[3])
    avg_meters['dice_std_atrium'].update(std_individual_labels[3])
    avg_meters['dice_lv_epi'].update(individual_labels[2])
    avg_meters['dice_std_lv_epi'].update(std_individual_labels[2])

    avg_meters['dice_mean'].update(dice_mean)

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice_lv_endo', avg_meters['dice_lv_endo'].val),
                        ('dice_std_lv_endo', avg_meters['dice_std_lv_endo'].val),
                        ('dice_atrium', avg_meters['dice_atrium'].val),
                        ('dice_std_atrium', avg_meters['dice_std_atrium'].val),
                        ('dice_lv_epi', avg_meters['dice_lv_epi'].val),
                        ('dice_std_lv_epi', avg_meters['dice_std_lv_epi'].val),
                        ('dice_mean', avg_meters['dice_mean'].val),
                        ])


def calc_case_dice_per_label(prediction, target, num_classes):
    return torch.nan_to_num(dice(prediction, target, average=None,
                                 num_classes=num_classes, ignore_index=0))


def calc_epoch_dice_per_label(batch_dice, funct_type):
    if funct_type == "mean":
        return np.mean(np.squeeze(batch_dice), axis=0)
    elif funct_type == "std":
        return torch.std(torch.stack(batch_dice), dim=0).detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown funct_type: {funct_type}, must be one of ['mean', 'std']")


def run_cycle(in_data_loader, num_classes, device, model, criterion, optimizer=None, current_epoch=None,
              total_epochs=None, desc="train", lr_scheduler=None):
    if desc.lower() == "train":
        model.train()
    elif desc.lower() == "validation" or desc.lower() == "test":
        model.eval()
    else:
        raise ValueError(f"Unknown desc: {desc}, must be one of ['train', 'validation', 'test']")

    with tqdm(in_data_loader, unit="batch") as tepoch:

        epoch_dice_individual_labels = []

        for target_img_batch, target_label_batch in tepoch:
            tepoch.set_description(f"Epoch {current_epoch}/{total_epochs} {desc}")
            target_img_batch, target_label_batch = target_img_batch.to(device), target_label_batch.to(device)
            target_label_batch = torch.nn.functional.one_hot(target_label_batch.long(), num_classes).permute(0, 3, 1, 2)

            if desc.lower() == "train":
                optimizer.zero_grad()

            target_pred_batch = model(target_img_batch)

            batch_loss = criterion(target_pred_batch, target_label_batch.float())
            if desc.lower() == "train":
                batch_loss.backward()
                optimizer.step()
                lr_scheduler.step()

            case_dice_individual_labels = []
            for case in range(target_pred_batch.shape[0]):  # looping over each case in batch to calculate metrics
                target_pred_case = torch.argmax(target_pred_batch[case, ...], dim=0)
                target_label_case = torch.argmax(target_label_batch[case, ...], dim=0)

                target_pred_epi_case = torch.clone(target_pred_case)
                target_epi = torch.clone(target_label_case)

                target_pred_epi_case[target_pred_epi_case == 1] = 2  # change labels to create epi mask
                target_epi[target_epi == 1] = 2

                dice_array = calc_case_dice_per_label(target_pred_case, target_label_case, num_classes)
                dice_array[2] = calc_case_dice_per_label(target_pred_epi_case, target_epi, num_classes)[2]  # epi dice
                case_dice_individual_labels += [dice_array.detach().cpu().numpy()]

            if len(epoch_dice_individual_labels) == 0:
                epoch_dice_individual_labels = case_dice_individual_labels
            else:
                epoch_dice_individual_labels = np.vstack((epoch_dice_individual_labels, case_dice_individual_labels))

            tepoch.set_postfix(loss=batch_loss.item())

    per_label_dice_mean = np.mean(epoch_dice_individual_labels, axis=0)
    per_label_dice_std = np.std(epoch_dice_individual_labels, axis=0)
    dice_mean = np.mean(per_label_dice_mean[1:])

    return calc_dice_metrics_heart(per_label_dice_mean,
                                   per_label_dice_std,
                                   batch_loss,
                                   dice_mean)

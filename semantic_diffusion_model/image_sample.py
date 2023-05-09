"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from argparse import ArgumentParser

import deepspeed
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision as tv
from PIL import Image
from skimage.color import label2rgb
from skimage.feature import canny

from config import cfg
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    create_model_and_diffusion,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Semantic Diffusion Model')
    parser.add_argument('--datadir',
                        default=cfg.DATASETS.DATADIR)
    parser.add_argument('--savedir',
                        default=cfg.DATASETS.SAVEDIR)
    parser.add_argument('--dataset_mode',
                        default=cfg.DATASETS.DATASET_MODE,
                        type=str)
    parser.add_argument('--learn_sigma',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.DIFFUSION.LEARN_SIGMA)
    parser.add_argument('--noise_schedule',
                        default=cfg.TRAIN.DIFFUSION.NOISE_SCHEDULE,
                        type=str)
    parser.add_argument('--timestep_respacing',
                        default=cfg.TRAIN.DIFFUSION.TIMESTEP_RESPACING,
                        type=str)
    parser.add_argument('--use_kl',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=cfg.TRAIN.DIFFUSION.USE_KL)
    parser.add_argument('--predict_xstart',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=cfg.TRAIN.DIFFUSION.PREDICT_XSTART)
    parser.add_argument('--rescale_timesteps',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=cfg.TRAIN.DIFFUSION.RESCALE_TIMESTEPS)
    parser.add_argument('--rescale_learned_sigmas',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=cfg.TRAIN.DIFFUSION.RESCALE_LEARNED_SIGMAS)
    parser.add_argument('--img_size',
                        default=cfg.TRAIN.IMG_SIZE,
                        type=int)
    parser.add_argument('--num_classes',
                        default=cfg.TRAIN.NUM_CLASSES,
                        type=int)
    parser.add_argument('--lr',
                        default=cfg.TRAIN.LR,
                        type=float)
    parser.add_argument('--attention_resolutions',
                        default=cfg.TRAIN.ATTENTION_RESOLUTIONS,
                        type=str)
    parser.add_argument('--channel_mult',
                        default=cfg.TRAIN.CHANNEL_MULT,
                        type=int)
    parser.add_argument('--dropout',
                        default=cfg.TRAIN.DROPOUT,
                        type=float)
    parser.add_argument('--diffusion_steps',
                        default=cfg.TRAIN.DIFFUSION_STEPS,
                        type=int)
    parser.add_argument('--schedule_sampler',
                        default=cfg.TRAIN.SCHEDULE_SAMPLER,
                        type=str)
    parser.add_argument('--num_channels',
                        default=cfg.TRAIN.NUM_CHANNELS,
                        type=int)
    parser.add_argument('--num_heads',
                        default=cfg.TRAIN.NUM_HEADS,
                        type=int)
    parser.add_argument('--num_heads_upsample',
                        default=cfg.TRAIN.NUM_HEADS_UPSAMPLE,
                        type=int)
    parser.add_argument('--num_head_channels',
                        default=cfg.TRAIN.NUM_HEAD_CHANNELS,
                        type=int)
    parser.add_argument('--num_res_blocks',
                        default=cfg.TRAIN.NUM_RES_BLOCKS,
                        type=int)
    parser.add_argument('--resblock_updown',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.RESBLOCK_UPDOWN)
    parser.add_argument('--use_scale_shift_norm',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.USE_SCALE_SHIFT_NORM)
    parser.add_argument('--use_checkpoint',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.USE_CHECKPOINT)
    parser.add_argument('--class_cond',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.CLASS_COND)
    parser.add_argument('--weight_decay',
                        default=cfg.TRAIN.WEIGHT_DECAY,
                        type=float)
    parser.add_argument('--lr_anneal_steps',
                        default=cfg.TRAIN.LR_ANNEAL_STEPS,
                        type=int)
    parser.add_argument('--batch_size_train',
                        default=cfg.TRAIN.BATCH_SIZE,
                        type=int)
    parser.add_argument('--microbatch',
                        default=cfg.TRAIN.MICROBATCH,
                        type=int)
    parser.add_argument('--ema_rate',
                        default=cfg.TRAIN.EMA_RATE,
                        type=str)
    parser.add_argument('--drop_rate',
                        default=cfg.TRAIN.DROP_RATE,
                        type=float)
    parser.add_argument('--log_interval',
                        default=cfg.TRAIN.LOG_INTERVAL,
                        type=int)
    parser.add_argument('--save_interval',
                        default=cfg.TRAIN.SAVE_INTERVAL,
                        type=int)
    parser.add_argument('--resume_checkpoint',
                        default=cfg.TRAIN.RESUME_CHECKPOINT)
    parser.add_argument('--use_fp16', type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.USE_FP16)
    parser.add_argument('--distributed_data_parallel',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL)
    parser.add_argument('--use_new_attention_order',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.USE_NEW_ATTENTION_ORDER)
    parser.add_argument('--fp16_scale_growth',
                        default=cfg.TRAIN.FP16_SCALE_GROWTH,
                        type=float)
    parser.add_argument('--num_workers',
                        default=cfg.TRAIN.NUM_WORKERS,
                        type=int)
    parser.add_argument('--no_instance',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.NO_INSTANCE)
    parser.add_argument('--deterministic_train',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.DETERMINISTIC)
    parser.add_argument('--random_crop',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.RANDOM_CROP)
    parser.add_argument('--random_flip',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.RANDOM_FLIP)
    parser.add_argument('--is_train',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TRAIN.IS_TRAIN)
    parser.add_argument('--s',
                        default=cfg.TEST.S,
                        type=float)
    parser.add_argument('--use_ddim',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TEST.USE_DDIM)
    parser.add_argument('--deterministic_test',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TEST.DETERMINISTIC)
    parser.add_argument('--inference_on_train',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TEST.INFERENCE_ON_TRAIN)
    parser.add_argument('--batch_size_test',
                        default=cfg.TEST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--clip_denoised',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=cfg.TEST.CLIP_DENOISED)
    parser.add_argument('--num_samples',
                        default=cfg.TEST.NUM_SAMPLES,
                        type=int)
    parser.add_argument('--results_dir',
                        default=cfg.TEST.RESULTS_DIR,
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = get_args_from_command_line()

    if args.datadir is not None:
        cfg.DATASETS.DATADIR = args.datadir
    if args.savedir is not None:
        cfg.TRAIN.SAVE_DIR = args.savedir
    if args.dataset_mode is not None:
        cfg.DATASETS.DATASET_MODE = args.dataset_mode
    if args.learn_sigma is not None:
        cfg.TRAIN.DIFFUSION.LEARN_SIGMA = args.learn_sigma
    if args.noise_schedule is not None:
        cfg.TRAIN.DIFFUSION.NOISE_SCHEDULE = args.noise_schedule
    if args.timestep_respacing is not None:
        cfg.TRAIN.DIFFUSION.TIMESTEP_RESPACING = args.timestep_respacing
    if args.use_kl is not None:
        cfg.TRAIN.DIFFUSION.USE_KL = args.use_kl
    if args.predict_xstart is not None:
        cfg.TRAIN.DIFFUSION.PREDICT_XSTART = args.predict_xstart
    if args.rescale_timesteps is not None:
        cfg.TRAIN.DIFFUSION.RESCALE_TIMESTEPS = args.rescale_timesteps
    if args.rescale_learned_sigmas is not None:
        cfg.TRAIN.DIFFUSION.RESCALE_LEARNED_SIGMAS = args.rescale_learned_sigmas
    if args.img_size is not None:
        cfg.TRAIN.IMG_SIZE = args.img_size
    if args.num_classes is not None:
        cfg.TRAIN.NUM_CLASSES = args.num_classes
    if args.lr is not None:
        cfg.TRAIN.LR = args.lr
    if args.attention_resolutions is not None:
        cfg.TRAIN.ATTENTION_RESOLUTIONS = args.attention_resolutions
    if args.channel_mult is not None:
        cfg.TRAIN.CHANNEL_MULT = args.channel_mult
    if args.dropout is not None:
        cfg.TRAIN.DROPOUT = args.dropout
    if args.diffusion_steps is not None:
        cfg.TRAIN.DIFFUSION.DIFFUSION_STEPS = args.diffusion_steps
    if args.schedule_sampler is not None:
        cfg.TRAIN.SCHEDULE_SAMPLER = args.schedule_sampler
    if args.num_channels is not None:
        cfg.TRAIN.NUM_CHANNELS = args.num_channels
    if args.num_heads is not None:
        cfg.TRAIN.NUM_HEADS = args.num_heads
    if args.num_heads_upsample is not None:
        cfg.TRAIN.NUM_HEADS_UPSAMPLE = args.num_heads_upsample
    if args.num_head_channels is not None:
        cfg.TRAIN.NUM_HEAD_CHANNELS = args.num_head_channels
    if args.num_res_blocks is not None:
        cfg.TRAIN.NUM_RES_BLOCKS = args.num_res_blocks
    if args.resblock_updown is not None:
        cfg.TRAIN.RESBLOCK_UPDOWN = args.resblock_updown
    if args.use_scale_shift_norm is not None:
        cfg.TRAIN.USE_SCALE_SHIFT_NORM = args.use_scale_shift_norm
    if args.use_checkpoint is not None:
        cfg.TRAIN.USE_CHECKPOINT = args.use_checkpoint
    if args.class_cond is not None:
        cfg.TRAIN.CLASS_COND = args.class_cond
    if args.weight_decay is not None:
        cfg.TRAIN.WEIGHT_DECAY = args.weight_decay
    if args.lr_anneal_steps is not None:
        cfg.TRAIN.LR_ANNEAL_STEPS = args.lr_anneal_steps
    if args.batch_size_train is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size_train
    if args.microbatch is not None:
        cfg.TRAIN.MICROBATCH = args.microbatch
    if args.ema_rate is not None:
        cfg.TRAIN.EMA_RATE = args.ema_rate
    if args.drop_rate is not None:
        cfg.TRAIN.DROP_RATE = args.drop_rate
    if args.log_interval is not None:
        cfg.TRAIN.LOG_INTERVAL = args.log_interval
    if args.save_interval is not None:
        cfg.TRAIN.SAVE_INTERVAL = args.save_interval
    if args.resume_checkpoint is not None:
        cfg.TRAIN.RESUME_CHECKPOINT = args.resume_checkpoint
    if args.use_fp16 is not None:
        cfg.TRAIN.USE_FP16 = args.use_fp16
    if args.distributed_data_parallel is not None:
        cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL = args.distributed_data_parallel
    if args.use_new_attention_order is not None:
        cfg.TRAIN.USE_NEW_ATTENTION_ORDER = args.use_new_attention_order
    if args.fp16_scale_growth is not None:
        cfg.TRAIN.FP16_SCALE_GROWTH = args.fp16_scale_growth
    if args.num_workers is not None:
        cfg.TRAIN.NUM_WORKERS = args.num_workers
    if args.no_instance is not None:
        cfg.TRAIN.NO_INSTANCE = args.no_instance
    if args.deterministic_train is not None:
        cfg.TRAIN.DETERMINISTIC = args.deterministic_train
    if args.random_crop is not None:
        cfg.TRAIN.RANDOM_CROP = args.random_crop
    if args.random_flip is not None:
        cfg.TRAIN.RANDOM_FLIP = args.random_flip
    if args.is_train is not None:
        cfg.TRAIN.IS_TRAIN = args.is_train
    if args.s is not None:
        cfg.TEST.S = args.s
    if args.use_ddim is not None:
        cfg.TEST.USE_DDIM = args.use_ddim
    if args.deterministic_test is not None:
        cfg.TEST.DETERMINISTIC = args.deterministic_test
    if args.inference_on_train is not None:
        cfg.TEST.INFERENCE_ON_TRAIN = args.inference_on_train
    if args.batch_size_test is not None:
        cfg.TEST.BATCH_SIZE = args.batch_size_test
    if args.clip_denoised is not None:
        cfg.TEST.CLIP_DENOISED = args.clip_denoised
    if args.num_samples is not None:
        cfg.TEST.NUM_SAMPLES = args.num_samples
    if args.results_dir is not None:
        cfg.TEST.RESULTS_DIR = args.results_dir

    deepspeed.init_distributed()
    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(cfg)

    model.load_state_dict(
        dist_util.load_state_dict(cfg.TRAIN.RESUME_CHECKPOINT)

    )
    if cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL:

        model.to(dist_util.dev())
    else:
        model.to('cuda')

    logger.log("creating data loader...")
    data = load_data(cfg)

    if cfg.TRAIN.USE_FP16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(cfg.TEST.RESULTS_DIR, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(cfg.TEST.RESULTS_DIR, 'labels')
    os.makedirs(label_path, exist_ok=True)
    visible_label_path = os.path.join(cfg.TEST.RESULTS_DIR, 'labels_visible')
    os.makedirs(visible_label_path, exist_ok=True)
    inference_path = os.path.join(cfg.TEST.RESULTS_DIR, 'samples')
    os.makedirs(inference_path, exist_ok=True)
    combined_path = os.path.join(cfg.TEST.RESULTS_DIR, 'combined')
    os.makedirs(combined_path, exist_ok=True)

    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        src_img = ((batch + 1.0) / 2.0).cuda()
        label_img = (cond['label_ori'].float())
        model_kwargs = preprocess_input(cond, num_classes=cfg.TRAIN.NUM_CLASSES)

        # set hyperparameter
        model_kwargs['s'] = cfg.TEST.S

        sample_fn = (
            diffusion.p_sample_loop if not cfg.TEST.USE_DDIM else diffusion.ddim_sample_loop
        )
        inference_img = sample_fn(
            model,
            (cfg.TEST.BATCH_SIZE, 3, src_img.shape[2], src_img.shape[3]),
            clip_denoised=cfg.TEST.CLIP_DENOISED,
            model_kwargs=model_kwargs,
            progress=True
        )

        inference_img = (inference_img + 1) / 2.0

        gathered_samples = [th.zeros_like(inference_img) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, inference_img)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(inference_img.shape[0]):
            tv.utils.save_image(src_img[j],
                                os.path.join(image_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))
            tv.utils.save_image(inference_img[j],
                                os.path.join(inference_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))
            tv.utils.save_image(label_img[j] / cfg.TRAIN.NUM_CLASSES,
                                os.path.join(visible_label_path,
                                             cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))

            label_save_img = Image.fromarray(label_img[j].cpu().detach().numpy()).convert('RGB')
            label_save_img.save(os.path.join(label_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))

            src_img_np = src_img[j].permute(1, 2, 0).detach().cpu().numpy()
            label_img_np = label_img[j].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()
            inference_img_np = (inference_img[j].permute(1, 2, 0).detach().cpu().numpy())
            inference_img_np = (inference_img_np - np.min(inference_img_np)) / np.ptp(inference_img_np)
            inference_img_np = (255 * (inference_img_np - np.min(inference_img_np)) / np.ptp(inference_img_np)).astype(
                int)

            combined_imgs = generate_combined_imgs(src_img_np,
                                                   label_img_np.astype(np.int_),
                                                   inference_img_np)

            im = Image.fromarray(combined_imgs)
            im.save(os.path.join(combined_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))

        logger.log(f"created {len(all_samples) * cfg.TEST.BATCH_SIZE} samples")

        if len(all_samples) * cfg.TEST.BATCH_SIZE > cfg.TEST.NUM_SAMPLES:
            break

    dist.barrier()
    logger.log("sampling complete")


def generate_combined_imgs(src_in_img, label_in_img, inference_in_img):
    overlayed_label = label2rgb(label=label_in_img[:, :, 0], image=inference_in_img,
                                bg_label=0,
                                channel_axis=-1,
                                alpha=0.2, image_alpha=1)

    src_out_img = (src_in_img * 255).astype('uint8')
    overlayed_label = (overlayed_label * 255).astype('uint8')

    edges = canny(label_in_img[:, :, 0] / label_in_img[:, :, 0].max())
    edges = np.expand_dims(edges, axis=-1)
    edges = np.concatenate((edges, edges, edges), axis=-1) * 255
    edges[:, :, 2] = 0

    overlayed_edge_label = np.copy(inference_in_img)
    overlayed_edge_label[edges == 255] = 255

    combined_imgs = np.concatenate((src_out_img, inference_in_img, overlayed_label, overlayed_edge_label),
                                   axis=0).astype(
        np.uint8)

    return combined_imgs


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


if __name__ == "__main__":
    main()

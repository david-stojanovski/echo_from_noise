"""
Train a diffusion model on images.
"""

import argparse
import json
import os
from argparse import ArgumentParser

import deepspeed

from config import cfg
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop


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
                        default=cfg.DATASETS.DATADIR,
                        type=str)
    parser.add_argument('--savedir',
                        default=cfg.DATASETS.SAVEDIR,
                        type=str)
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
    logger.configure(save_dir=cfg.DATASETS.SAVEDIR)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(cfg)
    if cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL:

        model.to(dist_util.dev())
    else:
        model.to('cuda')

    schedule_sampler = create_named_schedule_sampler(cfg.TRAIN.SCHEDULE_SAMPLER, diffusion)

    logger.log("creating data loader...")
    data = load_data(cfg)
    with open(os.path.join(cfg.DATASETS.SAVEDIR, 'train_test_config.json'), 'w') as fp:
        json.dump(cfg, fp, indent=4)
        fp.close()

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=cfg.TRAIN.NUM_CLASSES,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        microbatch=cfg.TRAIN.MICROBATCH,
        lr=cfg.TRAIN.LR,
        ema_rate=cfg.TRAIN.EMA_RATE,
        drop_rate=cfg.TRAIN.DROP_RATE,
        log_interval=cfg.TRAIN.LOG_INTERVAL,
        save_interval=cfg.TRAIN.SAVE_INTERVAL,
        resume_checkpoint=cfg.TRAIN.RESUME_CHECKPOINT,
        use_fp16=cfg.TRAIN.USE_FP16,
        fp16_scale_growth=cfg.TRAIN.FP16_SCALE_GROWTH,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        lr_anneal_steps=cfg.TRAIN.LR_ANNEAL_STEPS,
    ).run_loop()


if __name__ == "__main__":
    main()

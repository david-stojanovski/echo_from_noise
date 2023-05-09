import argparse

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel


def create_model_and_diffusion(cfg):
    model = create_model(
        image_size=cfg.TRAIN.IMG_SIZE,
        num_classes=cfg.TRAIN.NUM_CLASSES,
        num_channels=cfg.TRAIN.NUM_CHANNELS,
        num_res_blocks=cfg.TRAIN.NUM_RES_BLOCKS,
        channel_mult=cfg.TRAIN.CHANNEL_MULT,
        learn_sigma=cfg.TRAIN.DIFFUSION.LEARN_SIGMA,
        class_cond=cfg.TRAIN.CLASS_COND,
        use_checkpoint=cfg.TRAIN.USE_CHECKPOINT,
        attention_resolutions=cfg.TRAIN.ATTENTION_RESOLUTIONS,
        num_heads=cfg.TRAIN.NUM_HEADS,
        num_head_channels=cfg.TRAIN.NUM_HEAD_CHANNELS,
        num_heads_upsample=cfg.TRAIN.NUM_HEADS_UPSAMPLE,
        use_scale_shift_norm=cfg.TRAIN.USE_SCALE_SHIFT_NORM,
        dropout=cfg.TRAIN.DROPOUT,
        resblock_updown=cfg.TRAIN.RESBLOCK_UPDOWN,
        use_fp16=cfg.TRAIN.USE_FP16,
        use_new_attention_order=cfg.TRAIN.USE_NEW_ATTENTION_ORDER,
        no_instance=cfg.TRAIN.NO_INSTANCE,
    )
    diffusion = create_gaussian_diffusion(
        steps=cfg.TRAIN.DIFFUSION_STEPS,
        learn_sigma=cfg.TRAIN.DIFFUSION.LEARN_SIGMA,
        noise_schedule=cfg.TRAIN.DIFFUSION.NOISE_SCHEDULE,
        use_kl=cfg.TRAIN.DIFFUSION.USE_KL,
        predict_xstart=cfg.TRAIN.DIFFUSION.PREDICT_XSTART,
        rescale_timesteps=cfg.TRAIN.DIFFUSION.RESCALE_TIMESTEPS,
        rescale_learned_sigmas=cfg.TRAIN.DIFFUSION.RESCALE_LEARNED_SIGMAS,
        timestep_respacing=cfg.TRAIN.DIFFUSION.TIMESTEP_RESPACING,
    )
    return model, diffusion


def create_model(
        image_size,
        num_classes,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        no_instance=False,
):
    if channel_mult is None:
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    num_classes = num_classes if no_instance else num_classes + 1

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

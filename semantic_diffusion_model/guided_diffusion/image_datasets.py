import glob
import math
import os
import random

import blobfile as bf
import numpy as np
from PIL import Image
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset


def load_data(cfg):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    """
    if not cfg.DATASETS.DATADIR:
        raise ValueError("unspecified data directory")

    if cfg.DATASETS.DATASET_MODE == 'cityscapes':
        all_files = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'leftImg8bit', 'train' if cfg.TRAIN.IS_TRAIN else 'val'))
        labels_file = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'gtFine', 'train' if cfg.TRAIN.IS_TRAIN else 'val'))
        classes = [x for x in labels_file if x.endswith('_labelIds.png')]
        instances = [x for x in labels_file if x.endswith('_instanceIds.png')]
    elif cfg.DATASETS.DATASET_MODE == 'ade20k':
        all_files = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'images', 'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
        classes = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'annotations', 'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
        instances = None

    elif cfg.DATASETS.DATASET_MODE == 'camus':
        if cfg.TEST.INFERENCE_ON_TRAIN and not cfg.TRAIN.IS_TRAIN:  # inference on train in one go to make synthetic image generation easier
            all_files = glob.glob(os.path.join(cfg.DATASETS.DATADIR, 'images', 'training', '*.png'))
            all_files = all_files + glob.glob(os.path.join(cfg.DATASETS.DATADIR, 'images', 'validation', '*.png'))
            classes = glob.glob(os.path.join(cfg.DATASETS.DATADIR, 'sector_annotations', 'training', '*.png'))
            classes = classes + glob.glob(
                os.path.join(cfg.DATASETS.DATADIR, 'sector_annotations', 'validation', '*.png'))
            instances = None
        else:
            all_files = _list_image_files_recursively(
                os.path.join(cfg.DATASETS.DATADIR, 'images', 'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
            classes = _list_image_files_recursively(
                os.path.join(cfg.DATASETS.DATADIR, 'sector_annotations',
                             'training' if cfg.TRAIN.IS_TRAIN else 'validation'))
            instances = None
    elif cfg.DATASETS.DATASET_MODE == 'celeba':
        # The edge is computed by the instances.
        # However, the edge get from the labels and the instances are the same on CelebA.
        # You can take either as instance input
        all_files = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'train' if cfg.TRAIN.IS_TRAIN else 'test', 'images'))
        classes = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'train' if cfg.TRAIN.IS_TRAIN else 'test', 'labels'))
        instances = _list_image_files_recursively(
            os.path.join(cfg.DATASETS.DATADIR, 'train' if cfg.TRAIN.IS_TRAIN else 'test', 'labels'))
    else:
        raise NotImplementedError('{} not implemented'.format(cfg.DATASETS.DATASET_MODE))

    print("Len of Dataset:", len(all_files))

    dataset = ImageDataset(
        cfg.DATASETS.DATASET_MODE,
        cfg.TRAIN.IMG_SIZE,
        all_files,
        classes=classes,
        instances=instances,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=cfg.TRAIN.RANDOM_CROP,
        random_flip=cfg.TRAIN.RANDOM_FLIP,
        is_train=cfg.TRAIN.IS_TRAIN
    )

    if cfg.TRAIN.IS_TRAIN:
        batch_size = cfg.TRAIN.BATCH_SIZE
        if cfg.TRAIN.DETERMINISTIC:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
            )
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        if cfg.TEST.DETERMINISTIC:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
            )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
            self,
            dataset_mode,
            resolution,
            image_paths,
            classes=None,
            instances=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = None if instances is None else instances[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        out_dict = {}
        class_path = self.local_classes[idx]
        with bf.BlobFile(class_path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        pil_class = pil_class.convert("L")

        if self.local_instances is not None:
            instance_path = self.local_instances[idx]  # DEBUG: from classes to instances, may affect CelebA
            with bf.BlobFile(instance_path, "rb") as f:
                pil_instance = Image.open(f)
                pil_instance.load()
            pil_instance = pil_instance.convert("L")
        else:
            pil_instance = None

        if self.dataset_mode == 'cityscapes':
            arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution)

        if self.dataset_mode == 'camus':
            arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution,
                                                            keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None

        arr_image = arr_image.astype(np.float32) / 127.5 - 1

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()

        if self.dataset_mode == 'ade20k':
            arr_class = arr_class - 1
            arr_class[arr_class == 255] = 150
        elif self.dataset_mode == 'coco':
            arr_class[arr_class == 255] = 182

        out_dict['label'] = arr_class[None,]

        if arr_instance is not None:
            out_dict['instance'] = arr_instance[None,]

        return np.transpose(arr_image, [2, 0, 1]), out_dict


def resize_arr(pil_list, image_size, keep_aspect=True):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if keep_aspect:
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    return arr_image, arr_class, arr_instance


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_instance[crop_y: crop_y + image_size, crop_x: crop_x + image_size] if arr_instance is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_instance[crop_y: crop_y + image_size, crop_x: crop_x + image_size] if arr_instance is not None else None

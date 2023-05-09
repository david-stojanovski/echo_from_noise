import random

import numpy as np
import torch
import torchvision.transforms.functional as TF


class Compose(object):
    """ Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rendering_images, bounding_box=None):
        for t in self.transforms:
            if t.__class__.__name__ == 'RandomCrop' or t.__class__.__name__ == 'CenterCrop':
                rendering_images = t(rendering_images, bounding_box)
            else:
                rendering_images = t(rendering_images)

        return rendering_images


class GammaAdjust(object):
    """Randomly adjust gamma of the input image within a range"""

    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def __call__(self, x):
        gamma_val = random.uniform(*self.gamma_range)
        return TF.adjust_gamma(x, gamma_val)


class ToTensor(object):
    """
    Convert a PIL Image or numpy.ndarray to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        array = np.moveaxis(rendering_images, -1, 0)

        # handle numpy array
        tensor = torch.from_numpy(array)

        # put it from HWC to CHW format
        return tensor.float()


class TensorSqueeze(object):

    def __call__(self, rendering_images):
        return torch.squeeze(rendering_images, dim=0)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        rendering_images -= self.mean
        rendering_images /= self.std

        return rendering_images


class RemoveLabel(object):
    def __init__(self, label):
        self.label = label

    def __call__(self, rendering_images):
        assert (isinstance(rendering_images, np.ndarray))
        rendering_images[rendering_images == self.label] = 2
        rendering_images[rendering_images > self.label] -= 1
        return rendering_images


class Rescale2Dense(object):
    def __init__(self):
        pass

    def __call__(self, rendering_images):
        out_tensor = torch.clone(rendering_images)
        unique_labels = torch.unique(rendering_images)
        for idx, uniq_label in enumerate(unique_labels):
            out_tensor[rendering_images == uniq_label] = idx
        return out_tensor

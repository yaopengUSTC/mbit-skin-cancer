import random

import numpy as np
from skimage.filters import gaussian
import torch
from PIL import Image
from .color_constancy import color_constancy


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class MaskToTensor_uint8(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.uint8))


class MaskToTensor_float(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


class FreeScale(object):
    """Resize the image to a fixed size, and keep the horizontal and vertical ratio unchanged
        size：(h, w), the values to which the sides of the image is resized
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size  # size: (h, w)
        self.interpolation = interpolation

    def __call__(self, img):
        size_y = self.size[0]
        size_x = self.size[1]
        scale_y = img.size[1] / size_y
        scale_x = img.size[0] / size_x
        if scale_y < scale_x:  # select the smaller value
            size_x = int(img.size[0] / scale_y)
        else:
            size_y = int(img.size[1] / scale_x)
        return img.resize((size_x, size_y), self.interpolation)


class RandomCropInRate(object):
    """ random crop
        nsize: crop size
        rand_rate: The allowed region close to the center of the image for random cropping. (value: 0.7-1.0)
    """
    def __init__(self, nsize, rand_rate=(1.0, 1.0)):
        self.nsize = nsize
        self.rand_rate = rand_rate  # rand_rate: (l, s)

    def __call__(self, image):
        image_height = image.size[1]
        image_width = image.size[0]
        new_height = self.nsize[0]
        new_width = self.nsize[1]

        if image_width >= image_height:
            x_l = int(image_width * (1.0 - self.rand_rate[0]) / 2)
            x_r = int(image_width - x_l) - new_width
            y_l = int(image_height * (1.0 - self.rand_rate[1]) / 2)
            y_r = int(image_height - y_l) - new_height
        else:
            x_l = int(image_width * (1.0 - self.rand_rate[1]) / 2)
            x_r = int(image_width - x_l) - new_width
            y_l = int(image_height * (1.0 - self.rand_rate[0]) / 2)
            y_r = int(image_height - y_l) - new_height
        if x_r <= x_l or y_r <= y_l:
            raise ValueError('Invalid rand_rate: {}'.format(self.rand_rate))

        if 0 < new_height < image_height:
            start_h = random.randint(y_l, y_r)
        else:
            start_h = 0
            new_height = image_height
        if 0 < new_width < image_width:
            start_w = random.randint(x_l, x_r)
        else:
            start_w = 0
            new_width = image_width
        image = np.array(image)
        image = image[start_h:start_h + new_height, start_w:start_w + new_width, :]
        return Image.fromarray(image.astype(np.uint8))


class FlipChannels(object):
    def __call__(self, img):
        img = np.array(img)[:, :, ::-1]
        return Image.fromarray(img.astype(np.uint8))


class RandomGaussianBlur(object):
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


class NormalizePerImage(object):
    def __call__(self, tensor):
        """
        Normalize with the mean and variance of each image, not all images'
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not (torch.is_tensor(tensor) and tensor.ndimension() == 3):
            raise TypeError('tensor is not a torch image.')

        mean = torch.mean(tensor, (1, 2))
        for t, m in zip(tensor, mean):
            t.sub_(m)
        return tensor


class ColorConstancy(object):
    """ color constancy operation """
    def __init__(self, power=6, gamma=None):
        self.power = power
        self.gamma = gamma

    def __call__(self, img):
        img = color_constancy(np.array(img), self.power, self.gamma)
        return Image.fromarray(img.astype(np.uint8))

# Modified RandAugment
# ==============================================================================
"""Modified RandAugment util file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
import numpy as np
import random
import torch
import cv2

# when the image is transformed (e.g. rotate), some parts of the image will lose the  pixel values and need to be filled
g_replace_value = [128, 128, 128]
# the grading of the magnitude of the transformation
g_grade_num = 10
# the upper bound of the magnitude of the transformation（1--20）
# (if it exceeds grade_num, the amplitude of the transformation can exceed the maximum allowed）
g_magnitude_value = 10
# 'Constant Magnitude Method' or 'Random Magnitude Method with Increasing Upper Bound'
g_magnitude_is_constant = False

# Image format: the format of image is pytorch's tensor of type uint8
# the color order is RGB (PIL format), or BGR (opencv format)
g_color_order = "RGB"


def policy_rand(probability=0.5, magnitude=5):
    """
        Origin RandAugment method. (compare with the reference paper, the transformations 'translate-x/y' are removed
         and replaced by subsequent random crop operation)
    """
    policy = {
        0: [[('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],
            [('Rotate', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)]]
    }
    return policy


def policy_v0_0(probability=0.7, magnitude=5):
    """ Randomly select one transformation from all transformations. """
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)], [('Vignetting', probability, magnitude)],
        # shape augment
            [('Rotate', probability, magnitude)], [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)], [('Scale', probability, magnitude)],
            [('Scale_xy_diff', probability, magnitude)], [('Lens_distortion', probability, magnitude)]]
    }
    return policy


def policy_v1_0(probability=0.7, magnitude=5):
    """ Randomly select two transformations from all transformations."""
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],    # , [('Invert', probability, magnitude)]
        # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
        # color augment
        1: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],    # [('Invert', probability, magnitude)]
        # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
    }
    return policy


def policy_v1_1(probability=0.7, magnitude=5):
    """ Randomly select one transformation from {color} transformations,
        and then randomly select one transformation from {shape} transformations."""
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)]],
        # shape augment
        1: [[('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]]
    }
    return policy


def policy_v2_0(probability=0.7, magnitude=5):
    """ Randomly select three transformations from all transformations"""
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],    # [('Invert', probability, magnitude)]
            # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
        # color augment
        1: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],    # [('Invert', probability, magnitude)]
            # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
        # color augment
        2: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)],
            [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)],
            [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)],
            [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)],
            # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],
            # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
    }
    return policy


def policy_v2_1(probability=0.7, magnitude=5):
    """Randomly select one transformation from {color} transformations,
        and then randomly select two transformations from {shape} transformations."""
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)], [('Vignetting', probability, magnitude)]],
        # shape augment
        1: [[('Rotate', probability, magnitude)], [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)],
            [('Lens_distortion', probability, magnitude)]],
        2: [[('Rotate', probability, magnitude)], [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)],
            [('Lens_distortion', probability, magnitude)]]
    }
    return policy


def policy_v2_2(probability=0.7, magnitude=5):
    """Randomly select two transformations from {color} transformations,
        and then randomly select one transformation from {shape} transformations."""
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)], [('Vignetting', probability, magnitude)]],
        # color augment
        1: [[('Mixup', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)], [('Vignetting', probability, magnitude)]],
        2: [[('Rotate', probability, magnitude)], [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)],
            [('Lens_distortion', probability, magnitude)]]
    }
    return policy


def policy_v3_0(probability=0.7, magnitude=5):
    """ Randomly select four transformations from all transformations"""
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],    # , [('Invert', probability, magnitude)]
            # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
        # color augment
        1: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],    # , [('Invert', probability, magnitude)]
            # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
        # color augment
        2: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)],
            [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)],
            [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)],
            [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)],
            # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],
            # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
        # color augment
        3: [[('Mixup', probability, magnitude)], [('Vignetting', probability, magnitude)],
            [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)],
            [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)],
            [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)],
            # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)],
            # shape augment
            [('Rotate', probability, magnitude)], [('Lens_distortion', probability, magnitude)],
            [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)]],
    }
    return policy


def policy_v3_1(probability=0.7, magnitude=5):
    """Randomly select two transformations from {color} transformations,
        and then randomly select two transformations from {shape} transformations."""
    policy = {
        # color augment
        0: [[('Mixup', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)], [('Vignetting', probability, magnitude)]],
        1: [[('Mixup', probability, magnitude)], [('Gaussian_noise', probability, magnitude)],
            [('Saturation', probability, magnitude)], [('Contrast', probability, magnitude)], [('Brightness', probability, magnitude)],
            [('Sharpness', probability, magnitude)], [('Color_casting', probability, magnitude)], [('Equalize_YUV', probability, magnitude)],
            [('Posterize', probability, magnitude)], [('AutoContrast', probability, magnitude)], # [('SolarizeAdd', probability, magnitude)],
            [('Solarize', probability, magnitude)], [('Equalize', probability, magnitude)], [('Vignetting', probability, magnitude)]],
        # shape augment
        2: [[('Rotate', probability, magnitude)], [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)],
            [('Lens_distortion', probability, magnitude)]],
        3: [[('Rotate', probability, magnitude)], [('Flip', probability, magnitude)], [('Cutout', probability, magnitude)],
            [('Shear_x', probability, magnitude)], [('Shear_y', probability, magnitude)],
            [('Scale', probability, magnitude)], [('Scale_xy_diff', probability, magnitude)],
            [('Lens_distortion', probability, magnitude)]]
    }
    return policy


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.

    Returns:
      A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = image1.float()
    image2 = image2.float()

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp.type(torch.uint8)

    # We need to clip and then cast.
    temp = torch.clamp(temp, 0.0, 255.0)
    return temp.type(torch.uint8)


def mixup(image1, image_bg):
    """ mixup the corresponding pixels of image 1 and image 2. """
    _max = 0.2      # mixed intensity：0.0--0.2,
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    height, width = image1.shape[:2]
    image2 = cv2.resize(image_bg.numpy(), (width, height))
    image2 = torch.from_numpy(np.array(image2, dtype=np.uint8))
    return blend(image1, image2, factor)


def gaussian_noise(image):
    """ add Gaussian noise to the image. """
    _max = 0.2      # noise intensity：0.0--0.2,
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    size = tuple(image.shape)
    # rand = np.random.uniform(-50, 50, size)    # Random noise
    rand = np.random.normal(0, 50, size)         # Gaussian noise

    image1 = image.float().numpy() + rand * factor
    image1 = torch.from_numpy(image1)
    image1 = torch.clamp(image1, 0.0, 255.0)
    return image1.type(torch.uint8)


def cutout(image):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
      image: An image Tensor of type uint8.

    Returns:
      An image Tensor that is of type uint8.
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    # the range of the size of cutout: 0--50 pixel
    _max = 50
    _min = 0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    pad_size = round(factor)
    replace = tuple(g_replace_value)

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = random.randint(int(image_height * 0.2), int(image_height * 0.8))
    cutout_center_width = random.randint(int(image_width * 0.2), int(image_width * 0.8))

    lower_pad = max(0, cutout_center_height - pad_size)
    upper_pad = min(image_height, cutout_center_height + pad_size)
    left_pad = max(0, cutout_center_width - pad_size)
    right_pad = min(image_width, cutout_center_width + pad_size)

    image[lower_pad:upper_pad, left_pad:right_pad, 0] = replace[0]
    image[lower_pad:upper_pad, left_pad:right_pad, 1] = replace[1]
    image[lower_pad:upper_pad, left_pad:right_pad, 2] = replace[2]
    return image


def solarize(image):
    """ For each pixel in the image, select the pixel
        if the value is less than the threshold.
        Otherwise, subtract 255 from the pixel. """
    _min = 128
    _max = 255
    p = (_min - _max) / g_grade_num
    if g_magnitude_is_constant:
        factor = _max + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_max + p * g_magnitude_value, _max)       # magnitude is random
    threshold = round(factor)
    # threshold = random.randint(100, 250)
    return torch.where(image < threshold, image, 255 - image)


def solarize_add(image, threshold=128):
    """ For each pixel in the image less than threshold
        we add 'addition' amount to it and then clip the
        pixel value to be between 0 and 255. The value
        of 'addition' is between -128 and 128. """
    _max = 128
    _min = 1
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = index * factor
    addition = round(factor)
    added_image = image.int() + addition
    added_image = torch.clamp(added_image, 0, 255).byte()
    return torch.where(image < threshold, added_image, image)


def saturation(image):
    """ change the saturation of the image """
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    gray_img = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2GRAY)
    degenerate = torch.from_numpy(np.array(gray_img, dtype=np.uint8))  # h×w
    degenerate = degenerate.unsqueeze(-1)  # h*w*1
    degenerate = degenerate.repeat(1, 1, 3)  # h×w×3
    return blend(degenerate, image, factor)


def contrast(image):
    """ change the contrast of the image """
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    # factor = random.uniform(0.6, 1.4)
    gray_img = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2GRAY)
    mean = np.mean(np.array(gray_img))  # get the mean value of the gray image.
    degenerate = torch.full(image.shape, mean).byte()
    return blend(degenerate, image, factor)


def brightness(image):
    """ change the brightness of the image """
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    degenerate = torch.zeros(image.shape).byte()
    return blend(degenerate, image, factor)


def scale(image):
    """scale the image, and the width and height of the image are scaled in the same proportion.
        if it is reduced too much, the width or height may be smaller than the minimum input
        size required by the CNN model, and an error will occur.
    """
    _max = 0.2
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    height, width = image.shape[:2]
    new_height = round(height * factor)
    new_width = round(width * factor)

    img_scale = cv2.resize(image.numpy(), (new_width, new_height))
    img_scale = torch.from_numpy(np.array(img_scale, dtype=np.uint8))
    return torch.from_numpy(np.array(img_scale, dtype=np.uint8))


def scale_xy_diff(image):
    """scale the image, and the width and height of the image are scaled in the different proportion.
        if it is reduced too much, the width or height may be smaller than the minimum input
        size required by the CNN model, and an error will occur.
    """
    _max = 0.2
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor_x = 1.0 + index * factor
    if g_magnitude_is_constant:
        factor_y = 1.0
    else:
        factor_y = random.uniform(0.8, 1.2)
    height, width = image.shape[:2]
    new_height = round(height * factor_y)
    new_width = round(width * factor_x)

    img_scale = cv2.resize(image.numpy(), (new_width, new_height))
    img_scale = torch.from_numpy(np.array(img_scale, dtype=np.uint8))
    return torch.from_numpy(np.array(img_scale, dtype=np.uint8))


def shear_x(image):
    """the image is sheared in the horizontal direction"""
    _max = 15.0
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    degrees = index * factor
    replace = tuple(g_replace_value)
    height, width = image.shape[:2]

    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians
    x_move = (height / 2) * math.tan(radians)
    points1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    points2 = np.float32([[x_move, 0], [width + x_move, 0], [width - x_move, height], [-x_move, height]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    img_shear = cv2.warpPerspective(image.numpy(), matrix, (width, height), borderValue=replace)
    return torch.from_numpy(np.array(img_shear, dtype=np.uint8))


def shear_y(image):
    """the image is sheared in the vertical direction"""
    _max = 15.0
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    degrees = index * factor
    replace = tuple(g_replace_value)
    height, width = image.shape[:2]

    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians
    y_move = (width / 2) * math.tan(radians)
    points1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    points2 = np.float32([[0, y_move], [width, -y_move], [width, height - y_move], [0, height + y_move]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    img_shear = cv2.warpPerspective(image.numpy(), matrix, (width, height), borderValue=replace)
    return torch.from_numpy(np.array(img_shear, dtype=np.uint8))


def vignetting(image):
    """Compared with the center of the image, darken the periphery of the image.
    """
    _max = 0.6
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    height, width = image.shape[:2]

    center_x = random.uniform(-width / 10, width / 10)          # randomly determine the position of the center point of the vignetting
    center_y = random.uniform(-height / 10, height / 10)
    min_dist = width / 2.0 * np.random.uniform(0.3, 0.7)        # the distance from the starting position of vignetting to the center point

    # creat matrix of distance from the center on the two axis
    x, y = np.meshgrid(np.linspace(-width / 2 + center_x, width / 2 + center_x, width),
                       np.linspace(-height / 2 + center_y, height / 2 + center_y, height))
    x, y = np.abs(x), np.abs(y)
    z = np.sqrt(x**2 + y**2)

    # creat the vignette mask on the two axis
    z = (z - min_dist) / (np.max(z) - min_dist)
    z = np.clip(z, 0, 1)
    z = z**1.2              # change to non-linear
    z = z * factor
    z = torch.from_numpy(z)  # h×w
    z = z.unsqueeze(-1)  # h*w*1
    z = z.repeat(1, 1, 3)  # h×w×3

    image = image.float()
    image = image * (1.0 - z)
    image = torch.clamp(image, 0.0, 255.0)
    return image.type(torch.uint8)


def lens_distortion(image):
    """ simulate lens distortion to transform the image.
    """
    d_coef = np.array([0.15, 0.15, 0.1, 0.1])       # k1, k2, p1, p2
    _max = 0.6
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        d_factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        d_factor = np.random.uniform(_min, _min + p * g_magnitude_value, 4)      # magnitude is random
    d_factor = d_factor * (2 * (np.random.random(4) < 0.5) - 1)     # add sign
    d_coef = d_coef * d_factor

    height, width = image.shape[:2]
    # compute its diagonal
    f = (height**2 + width**2)**0.5
    # set the image projective to carrtesian dimension
    K = np.array([[f, 0, width / 2],
                  [0, f, height / 2],
                  [0, 0, 1]])
    # Generate new camera matrix from parameters
    M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (width, height), 0.5)
    # Generate look-up tables for remapping the camera image
    remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (width, height), cv2.CV_32FC2)
    # Remap the original image to a new image
    replace = tuple(g_replace_value)
    img = cv2.remap(image.numpy(), *remap, cv2.INTER_LINEAR, borderValue=replace)
    return torch.from_numpy(np.array(img, dtype=np.uint8))


def posterize(image):
    """Equivalent of PIL Posterize. change the low n-bits of each pixel of the image to 0"""
    _min = 0.0
    _max = 3.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    bits = round(factor)
    shift = bits
    img = image.byte() >> shift
    img = img << shift
    return img


def invert(image):
  """Inverts the image pixels."""
  return 255 - image


def rotate(image):
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
      image: An image Tensor of type uint8.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
      replace: A one or three value 1D tuple to fill empty pixels caused by
        the rotate operation.

    Returns:
      The rotated version of image.
    """
    _max = 40.0
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    degrees = index * factor
    replace = tuple(g_replace_value)
    height, width = image.shape[:2]
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    imgRotation = cv2.warpAffine(image.numpy(), matRotation, (width, height), borderValue=replace)
    return torch.from_numpy(np.array(imgRotation, dtype=np.uint8))


def autocontrast(image):
    """Implements Autocontrast function from PIL using TF ops.
    Args:
      image: A 3D uint8 tensor.

    Returns:
      The image after it has had autocontrast applied to it and will be of type
      uint8.
    """
    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = torch.min(image).float()
        hi = torch.max(image).float()

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.float() * scale + offset
            im = torch.clamp(im, 0.0, 255.0)
            return im.type(torch.uint8)

        return torch.where(hi > lo, scale_values(image), image)

    image = scale_channel(image)
    return image


def sharpness(image):
    """Implements Sharpness function from PIL using TF ops.
    """
    _max = 0.4
    _min = 0.0
    p = (_max - _min) / g_grade_num
    if g_magnitude_is_constant:
        factor = _min + p * g_magnitude_value     # magnitude is fixed
    else:
        factor = random.uniform(_min, _min + p * g_magnitude_value)       # magnitude is random
    index = random.sample([-1, 1], 1)[0]
    factor = 1.0 + index * factor
    orig_image = image
    image = image.float()
    # Make image 4D for conv operation.
    image = image.permute(2, 0, 1).contiguous()  # data from h*w*3 to 3*h*w
    image = image.unsqueeze(0)  # 1*3*h*w
    # smooth PIL Kernel.
    kernel = torch.tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=torch.float32)  # 3*3
    kernel = kernel / 13.  # normalize
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 1*1*3*3
    kernel = kernel.repeat(3, 1, 1, 1)  # 3*1*3*3

    conv = torch.nn.Conv2d(3, 3, 3, groups=3, bias=False)
    conv.weight.data = kernel
    degenerate = conv(image)
    degenerate = torch.clamp(degenerate, 0, 255).byte().squeeze(0)
    degenerate = degenerate.permute(1, 2, 0).contiguous()  # data form 3×h×w to h×w×3

    # For the borders of the resulting image, fill in the values of the
    # original image.
    result = torch.zeros(orig_image.shape).byte()
    result[:, :, :] = orig_image
    result[1:-1, 1:-1, :] = degenerate
    return blend(result, orig_image, factor)


def equalize(image):
    """Implements Equalize function from PIL using TF ops.
        For each color channel, implements Equalize function.
    """

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c].numpy()
        inhist = cv2.equalizeHist(im)
        return torch.from_numpy(np.array(inhist, dtype=np.uint8))

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0).unsqueeze(-1)
    s2 = scale_channel(image, 1).unsqueeze(-1)
    s3 = scale_channel(image, 2).unsqueeze(-1)
    image = torch.cat([s1, s2, s3], 2)
    return image


def equalize_YUV(image):
    """Implements Equalize function from PIL using TF ops.
        Transforms the image to YUV color space, and then only implements Equalize function on the brightness Y
    """
    img = image.numpy()
    if g_color_order == "RGB":        # PIL format
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:                           # Opencv format
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channels_yuv = cv2.split(img_yuv)
    channels_yuv[0] = cv2.equalizeHist(channels_yuv[0])
    channels = cv2.merge(channels_yuv)
    if g_color_order == "RGB":        # PIL format
        result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2RGB)
    else:                           # Opencv format
        result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    return torch.from_numpy(np.array(result, dtype=np.uint8))


def flip(image):
    """ Image is randomly flipped
    """
    # factor: 0:vertical; 1:horizontal; -1:diagonal mirror
    factor = random.randint(-1, 1)
    img = image.numpy()
    img = cv2.flip(img, factor)
    return torch.from_numpy(np.array(img, dtype=np.uint8))


def crop(image, need_rand=True, nsize=(224, 224), rand_rate=(1.0, 1.0)):
    """ random crop
        nsize: crop size
        need_rand: random crop or center crop
        rand_rate: The allowed region close to the center of the image for random cropping. (value: 0.7-1.0)
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    new_height = round(nsize[0])
    new_width = round(nsize[1])

    if image_width >= image_height:     # width is greater than height
        x_l = int(image_width * (1.0 - rand_rate[0]) / 2)
        x_r = int(image_width - x_l) - new_width
        y_l = int(image_height * (1.0 - rand_rate[1]) / 2)
        y_r = int(image_height - y_l) - new_height
    else:                               # width is smaller than height
        x_l = int(image_width * (1.0 - rand_rate[1]) / 2)
        x_r = int(image_width - x_l) - new_width
        y_l = int(image_height * (1.0 - rand_rate[0]) / 2)
        y_r = int(image_height - y_l) - new_height
    if x_r <= x_l or y_r <= y_l:
        raise ValueError('Invalid rand_rate: {}'.format(rand_rate))

    if 0 < new_height < image_height:
        if need_rand:
            start_h = random.randint(y_l, y_r)
        else:
            start_h = int((image_height - new_height) / 2)
    else:
        start_h = 0
        new_height = image_height
    if 0 < new_width < image_width:
        if need_rand:
            start_w = random.randint(x_l, x_r)
        else:
            start_w = int((image_width - new_width) / 2)
    else:
        start_w = 0
        new_width = image_width
    image = image[start_h:start_h + new_height, start_w:start_w + new_width, :]
    return image


def resize(image, min_size=256):
    """Resize the image to a fixed size, and keep the horizontal and vertical ratio unchanged
        min_size：the value to which the short side of the image is resized
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    if image_height < image_width:
        new_height = round(min_size)
        factor = min_size / image_height
        new_width = round(image_width * factor)
    else:
        new_width = round(min_size)
        factor = min_size / image_width
        new_height = round(image_height * factor)

    img_scale = cv2.resize(image.numpy(), (new_width, new_height))
    img_scale = torch.from_numpy(np.array(img_scale, dtype=np.uint8))
    return torch.from_numpy(np.array(img_scale, dtype=np.uint8))


def color_casting(image):
    """ Add a bias to a color channel in RGB
        For example, add a bias of 15 in the B color channel to each pixel, the image will be bluish.
    """
    prob_0 = random.randint(0, 2)
    for i in range(3):
        prob = random.randint(-1, 1)
        if prob_0 == i or prob == 1:
            _max = 30.0
            _min = 0.0
            p = (_max - _min) / g_grade_num
            if g_magnitude_is_constant:
                factor = _min + p * g_magnitude_value  # magnitude is fixed
            else:
                factor = random.uniform(_min, _min + p * g_magnitude_value)  # magnitude is random
            index = random.sample([-1, 1], 1)[0]
            bias = index * factor
            img = image[:, :, i].float()
            img = img + bias
            img = torch.clamp(img, 0.0, 255.0).type(torch.uint8)
            image[:, :, i] = img
    return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Equalize_YUV': equalize_YUV,
    'Rotate': rotate,
    'Shear_x': shear_x,
    'Shear_y': shear_y,
    'Scale': scale,
    'Scale_xy_diff': scale_xy_diff,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Saturation': saturation,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'Cutout': cutout,
    'Gaussian_noise': gaussian_noise,
    'Vignetting': vignetting,
    'Lens_distortion': lens_distortion,
    'Mixup': mixup,
    'Flip': flip,
    'Crop': crop,
    'Color_casting': color_casting,
    'Resize': resize,
    'Invert': invert,
}


def _parse_policy_info(image_bg, name, prob, magnitude):
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]

    args = ()
    # Add in image_bg arg if it is required for the function that is being called.
    if 'image_bg' in inspect.getargspec(func)[0]:
        # Make sure image_bg is the final argument
        assert 'image_bg' == inspect.getargspec(func)[0][-1]
        args = (image_bg,)

    return (func, prob, magnitude, args)


def _apply_func_with_prob(func, image, args, prob):
    """Apply `func` to image w/ `args` as input with probability `prob`."""

    # Apply the function with probability `prob`.
    should_apply_op = random.random() + prob
    if should_apply_op >= 1.0:
        augmented_image = func(image, *args)
        return augmented_image, func
    else:
        augmented_image = image
        return augmented_image, None


def select_and_apply_random_policy(policies, image, have_run_func=[]):
    """Select a random policy from `policies` and apply it to `image`."""
    while True:
        have_run = False
        policy_to_select = random.randint(0, len(policies) - 1)
        for (i, tf_policy) in enumerate(policies):
            if policy_to_select == i:
                for func, prob, magnitude, args in tf_policy:
                    if func in have_run_func:
                        have_run = True
                        break
                if not have_run:
                    for func, prob, magnitude, args in tf_policy:
                        global g_magnitude_value
                        g_magnitude_value = magnitude
                        image, run_func = _apply_func_with_prob(func, image, args, prob)
                        if run_func:
                            have_run_func.append(run_func)
        if not have_run:
            break
    return image, have_run_func


def build_and_apply_nas_policy(policies, image, image_bg=None, have_run_func=[]):
    """Build a policy from the given policies passed in and apply to image.

    Args:
        policies: list of lists of tuples in the form `(func, prob, level)`, `func`
            is a string name of the augmentation function, `prob` is the probability
            of applying the `func` operation, `level` is the input argument for
            `func`.
        image: pytorch Tensor that the resulting policy will be applied to.
        image_bg: as background noise, another image to be superimposed on 'image'.
        have_run_func: for the same image, the same transformation can only be used once.
    Returns:
        A version of image that now has data augmentation applied to it based on
        the `policies` pass into the function. Additionally, returns bboxes if
        a value for them is passed in that is not None
    """

    ag_policies = []
    for policy in policies:
        ag_policy = []
        # Link string name to the correct python function and make sure the correct
        # argument is passed into that function.
        for policy_info in policy:
            policy_info = list(policy_info)
            ag_policy.append(_parse_policy_info(image_bg, *policy_info))

        ag_policies.append(ag_policy)

    augmented_images, have_run_func = select_and_apply_random_policy(
        ag_policies, image, have_run_func)
    # then just return the images.
    return augmented_images, have_run_func


def distort_image_with_randaugment(image, augmentation_name, num_layers, replace=None, magnitude=10):
    """Applies the RandAugment policy to `image`.

    Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        augmentation_name: The name of the RandAugment policy to use. The available
            options are `rand`.
        num_layers: Integer, the number of augmentation transformations to apply sequentially to an image.
            Represented as (N) in the paper. Usually best values will be in the range [1, 3].
        replace: when the image is transformed, some parts of the image will lose the  pixel values and need to be filled
        magnitude:  the magnitude of the transformation（value: 1--20）
    Returns:
      The augmented image.
    """
    available_policies = {'rand': policy_rand}
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))

    if replace is not None:
        global g_replace_value
        if isinstance(replace, (int, float)):
            g_replace_value[0] = float(replace)
            g_replace_value[1] = float(replace)
            g_replace_value[2] = float(replace)
        elif isinstance(replace, (list, tuple)) and len(replace) == 3:
            g_replace_value[:] = list(replace)

    policy = available_policies[augmentation_name](0.5, magnitude)
    have_run_func = []
    for i in range(num_layers):
        image, have_run_func = build_and_apply_nas_policy(policy[0], image, None, have_run_func)

    return image


def distort_image_with_modified_randaugment(image, augmentation_name, image_bg, replace=None, probability=0.7, magnitude=10):
    """Applies the modified RandAugment policy to `image`.

    Args:
        image: `Tensor` of shape [height, width, 3] representing an image.
        augmentation_name: The name of the Modified RandAugment policy to use. The available
            options are `v0_0`, `v1_0`, `v1_1`, `v2_0`, `v2_1`, `v2_2`, `v3_0` and `v3_1`.
        image_bg: as background noise, another image to be superimposed on 'image'.
        replace: when the image is transformed, some parts of the image will lose the  pixel values and need to be filled
        probability: the probability of the transformation（0.1--0.9）
        magnitude:  the magnitude of the transformation（value: 1--20）
    Returns:
      The augmented image.
    """
    available_policies = {'v0_0': policy_v0_0,
                          'v1_0': policy_v1_0, 'v1_1': policy_v1_1,
                          'v2_0': policy_v2_0, 'v2_1': policy_v2_1, 'v2_2': policy_v2_2,
                          'v3_0': policy_v3_0, 'v3_1': policy_v3_1,
                          }
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))

    if replace is not None:
        global g_replace_value
        if isinstance(replace, (int, float)):
            g_replace_value[0] = float(replace)
            g_replace_value[1] = float(replace)
            g_replace_value[2] = float(replace)
        elif isinstance(replace, (list, tuple)) and len(replace) == 3:
            g_replace_value[:] = list(replace)

    policy = available_policies[augmentation_name](probability, magnitude)
    have_run_func = []
    for i in range(len(policy)):
        image, have_run_func = build_and_apply_nas_policy(policy[i], image, image_bg, have_run_func)

    return image


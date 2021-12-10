import json, os
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import numpy as np


def parse_args():
    '''
    Many images of ISIC 2019 have black borders or are too large. The black borders need to be removed,
    and the images need to be resize to a fixed size.
    example: python resize_image_ISIC_2019.py
                --csv_file ./dataset_files/ISIC_2019_Train.csv
                --root /SDG/work_image/image/skin_cancer/ISIC2019/ISIC_2019_Training_Input/
                --sp /home/yp/4T/jsons
                --small_path ./dataset_image/isic_2019/ISIC_2019_Train_Small/
    '''
    parser = argparse.ArgumentParser(description="Resize the images of ISIC 2019 dataset.")

    parser.add_argument(
        "--csv_file",
        help="csv file to be converted",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--root",
        help="root path to save image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sp",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )
    parser.add_argument(
        "--small_path",
        help="save path for small images",
        type=str,
        required=True,
        default="."
    )

    args = parser.parse_args()
    return args


def convert(args):
    csv_data = pd.read_csv(args.csv_file)

    print("Converting file {} ...".format(args.csv_file))

    for i in tqdm(range(len(csv_data))):
        image_id = csv_data.iloc[i]['image']
        derm_path = os.path.join(args.root, image_id + ".jpg")
        assert os.path.exists(derm_path), \
            'Image file does not exist: {}'.format(derm_path)

        img = Image.open(derm_path)
        min_size = min(img.size)
        croping = transforms.CenterCrop(min_size)
        img = croping(img)      # The image is cropped to a square based on the short side.

        # Remove the black borders
        threshold = 40
        img = np.array(img)

        ll = 1      # The most edge line may be bright, so start searching from the second line.
        while True:
            img_ll = img[ll:ll+1, :, :]
            if np.max(img_ll) < threshold:
                ll += 1
            else:
                break
        if ll > 1:
            img = img[ll:, :, :]

        ll = 1
        while True:
            img_ll = img[-1-ll:-ll, :, :]
            if np.max(img_ll) < threshold:
                ll += 1
            else:
                break
        if ll > 1:
            img = img[:-ll, :, :]

        ll = 1
        while True:
            img_ll = img[:, ll:ll+1, :]
            if np.max(img_ll) < threshold:
                ll += 1
            else:
                break
        if ll > 1:
            img = img[:, ll:, :]

        ll = 1
        while True:
            img_ll = img[:, -1-ll:-ll, :]
            if np.max(img_ll) < threshold:
                ll += 1
            else:
                break
        if ll > 1:
            img = img[:, :-ll, :]

        # Determine whether the four corners of the image are dark, if so, further cropping is needed.
        step = 10
        img_1 = img
        c_l_all, c_r_all, c_t_all, c_b_all = 0, 0, 0, 0
        while True:
            h, w = img.shape[0], img.shape[1]
            c_l, c_r, c_t, c_b = 0, 0, 0, 0
            img_l_t = img[1:step, 1:step, :]
            img_r_t = img[1:step, w-step:w-1, :]
            img_l_b = img[h-step:h-1, 1:step, :]
            img_r_b = img[h-step:h-1, w-step:w-1, :]
            if np.max(img_l_t) < threshold:
                c_l, c_t = step, step
            if np.max(img_r_t) < threshold:
                c_r, c_t = step, step
            if np.max(img_l_b) < threshold:
                c_l, c_b = step, step
            if np.max(img_r_b) < threshold:
                c_r, c_b = step, step

            if c_l > 0 or c_r > 0:
                img = img[c_t:h-c_b, c_l:w-c_r, :]
                c_l_all += c_l
                c_r_all += c_r
                c_t_all += c_t
                c_b_all += c_b
            else:
                break
        min_size = min(img.shape[0], img.shape[1])
        aug_size = int(min_size * 0.05)
        c_l_all = max(0, c_l_all - aug_size)
        c_r_all = max(0, c_r_all - aug_size)
        c_t_all = max(0, c_t_all - aug_size)
        c_b_all = max(0, c_b_all - aug_size)
        h, w = img_1.shape[0], img_1.shape[1]
        # A few images may be darker overall, so there is no need to crop.
        if c_l_all > h * 0.3 or c_r_all > h * 0.3 or c_t_all > h * 0.3 or c_b_all > h * 0.3:
            img = img_1
        else:
            img = img_1[c_t_all: h - c_b_all, c_l_all: w - c_r_all, :]

        img = Image.fromarray(img)
        min_size = min(img.size)
        croping = transforms.CenterCrop(min_size)
        img = croping(img)      # The image is cropped to a square based on the short side again.

        # the input image resize to a fix size
        resizing = transforms.Resize(450)
        img = resizing(img)
        # img.show()
        save_root = os.path.join(args.small_path, image_id + ".jpg")
        img.save(save_root, quality=95)

    print(" all small images are saved to {}".format(args.small_path))


if __name__ == "__main__":
    args = parse_args()
    convert(args)

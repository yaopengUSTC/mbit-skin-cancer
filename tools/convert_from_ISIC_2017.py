import json, os
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd
import ast
import torchvision.transforms as transforms


def parse_args():
    '''
    example: python convert_from_Derm_7pt.py
                --file ./dataset_files/ISIC_2017_Train.csv
                --root /SDG/work_image/image/skin_cancer/ISIC2017/ISIC-2017_Training_Data/
                --sp ../main/jsons/
                --small True
                --small_path ./dataset_image/isic_2017/train/
    '''
    parser = argparse.ArgumentParser(description="Get the json file for ISIC 2017 dataset.")

    parser.add_argument(
        "--file",
        help="origin csv file to be converted",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--root",
        help="root path to the images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sp",
        help="save path for converted file",
        type=str,
        required=False,
        default="."
    )
    # Many images in isic 2017 are too big, which causes the dataloader to process images very slowly.
    # So we can reduce the images to a fixed size in advance.
    parser.add_argument(
        "--small",
        help="reduce the images to small",
        type=ast.literal_eval,
        required=False,
        default=False
    )
    parser.add_argument(
        "--small_path",
        help="save path for small images",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args


def convert(args):
    csv_data = pd.read_csv(args.file)

    new_annos = []
    print("Converting file {} ...".format(args.file))

    for i in tqdm(range(len(csv_data))):
        image_id = csv_data.iloc[i]['image_id']
        label = int(csv_data.iloc[i]["melanoma"]) \
                + int(csv_data.iloc[i]["seborrheic_keratosis"]) * 2 \
                + int(csv_data.iloc[i]["nv"]) * 3 - 1
        derm_path = os.path.join(args.root, image_id + ".jpg")
        assert os.path.exists(derm_path), \
            'Image file does not exist: {}'.format(derm_path)
        img = Image.open(derm_path)

        if args.small:
            resizing = transforms.Resize(450)
            img = resizing(img)
            # img.show()
            derm_path = os.path.join(args.small_path, image_id + ".jpg")
            img.save(derm_path, quality=95)

        sizes = img.size
        derm_width = sizes[0]
        derm_height = sizes[1]

        new_annos.append({"image_id": image_id,
                          "category_id": label,
                          "derm_height": derm_height,
                          "derm_width": derm_width,
                          "derm_path": derm_path})
    num_classes = 3
    return {"annotations": new_annos,
            "num_classes": num_classes}


if __name__ == "__main__":
    args = parse_args()
    converted_annos = convert(args)
    if args.small:
        save_path = os.path.join(args.sp, "converted_" + os.path.splitext(os.path.split(args.file)[-1])[0] + "_Small.json")
    else:
        save_path = os.path.join(args.sp, "converted_" + os.path.splitext(os.path.split(args.file)[-1])[0] + ".json")
    print("Converted, Saveing converted file to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)


import json, os
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd
import pickle


def parse_args():
    '''
    Before running this code, the code "resize_image_ISIC_2019.py" needs to be run first.
    example: python convert_from_ISIC_2019.py
                --csv_file ./dataset_files/ISIC_2019_Train.csv
                --pkl_file ./dataset_files/ISIC_2019_5foldcv_indices.pkl
                --root ./dataset_image/isic_2019/ISIC_2019_Train_Small/
                --sp ../main/jsons/
    '''
    parser = argparse.ArgumentParser(description="Get the json file for ISIC 2019 dataset.")

    parser.add_argument(
        "--csv_file",
        help="csv file to be converted",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--pkl_file",
        help="pkl file to get the training and val index for cross validation",
        required=False,
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
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args


def convert(args):
    csv_data = pd.read_csv(args.csv_file)

    num_classes = 8
    if "Test" in args.csv_file:     # the test set of ISIC 2019 adds a class "UNK"
        num_classes = 9
    print("Converting file {} ...".format(args.csv_file))

    if args.pkl_file is not None:  # 5 fold cross validation
        assert "Train" in args.csv_file, 'cross validation is only for training set.'
        with open(args.pkl_file, 'rb') as f:
            indices = pickle.load(f)
            for j in range(len(indices['trainIndCV'])):
                train_idx = indices['trainIndCV'][j]
                val_idx = indices['valIndCV'][j]
                new_annos = []
                for ii in tqdm(range(len(train_idx))):
                    i = train_idx[ii]
                    image_id = csv_data.iloc[i]['image']
                    label = int(csv_data.iloc[i]["MEL"]) \
                            + int(csv_data.iloc[i]["NV"]) * 2 \
                            + int(csv_data.iloc[i]["BCC"]) * 3 \
                            + int(csv_data.iloc[i]["AK"]) * 4 \
                            + int(csv_data.iloc[i]["BKL"]) * 5 \
                            + int(csv_data.iloc[i]["DF"]) * 6 \
                            + int(csv_data.iloc[i]["VASC"]) * 7 \
                            + int(csv_data.iloc[i]["SCC"]) * 8 - 1

                    derm_path = os.path.join(args.root, image_id + ".jpg")
                    assert os.path.exists(derm_path), \
                        'Image file does not exist: {}'.format(derm_path)
                    sizes = Image.open(derm_path).size
                    derm_width = sizes[0]
                    derm_height = sizes[1]
                    new_annos.append({"image_id": image_id,
                                      "category_id": label,
                                      "derm_height": derm_height,
                                      "derm_width": derm_width,
                                      "derm_path": derm_path})
                converted_annos = {"annotations": new_annos,
                                   "num_classes": num_classes}
                save_path = os.path.join(args.sp,
                                         "converted_" + "ISIC_2019_Train_" + str(j) + ".json")
                print("Converted, Saving converted file to {}".format(save_path))
                with open(save_path, "w") as f:
                    json.dump(converted_annos, f)

                new_annos = []
                for ii in tqdm(range(len(val_idx))):
                    i = val_idx[ii]
                    image_id = csv_data.iloc[i]['image']
                    label = int(csv_data.iloc[i]["MEL"]) \
                            + int(csv_data.iloc[i]["NV"]) * 2 \
                            + int(csv_data.iloc[i]["BCC"]) * 3 \
                            + int(csv_data.iloc[i]["AK"]) * 4 \
                            + int(csv_data.iloc[i]["BKL"]) * 5 \
                            + int(csv_data.iloc[i]["DF"]) * 6 \
                            + int(csv_data.iloc[i]["VASC"]) * 7 \
                            + int(csv_data.iloc[i]["SCC"]) * 8 - 1
                    derm_path = os.path.join(args.root, image_id + ".jpg")
                    assert os.path.exists(derm_path), \
                        'Image file does not exist: {}'.format(derm_path)
                    sizes = Image.open(derm_path).size
                    derm_width = sizes[0]
                    derm_height = sizes[1]
                    new_annos.append({"image_id": image_id,
                                      "category_id": label,
                                      "derm_height": derm_height,
                                      "derm_width": derm_width,
                                      "derm_path": derm_path})
                converted_annos = {"annotations": new_annos,
                                   "num_classes": num_classes}
                save_path = os.path.join(args.sp,
                                         "converted_" + "ISIC_2019_Val_" + str(j) + ".json")
                print("Converted, Saving converted file to {}".format(save_path))
                with open(save_path, "w") as f:
                    json.dump(converted_annos, f)
    else:   # all images in the dataset are used
        annos = []
        for i in tqdm(range(len(csv_data))):
            image_id = csv_data.iloc[i]['image']
            if "Train" in args.csv_file or "Val" in args.csv_file:  # for train and validation
                label = int(csv_data.iloc[i]["MEL"]) \
                        + int(csv_data.iloc[i]["NV"]) * 2 \
                        + int(csv_data.iloc[i]["BCC"]) * 3 \
                        + int(csv_data.iloc[i]["AK"]) * 4 \
                        + int(csv_data.iloc[i]["BKL"]) * 5 \
                        + int(csv_data.iloc[i]["DF"]) * 6 \
                        + int(csv_data.iloc[i]["VASC"]) * 7 \
                        + int(csv_data.iloc[i]["SCC"]) * 8 - 1

            derm_path = os.path.join(args.root, image_id + ".jpg")
            assert os.path.exists(derm_path), \
                'Image file does not exist: {}'.format(derm_path)

            sizes = Image.open(derm_path).size
            derm_width = sizes[0]
            derm_height = sizes[1]
            if "Train" in args.csv_file or "Val" in args.csv_file:
                sample = {"image_id": image_id,
                          "category_id": label,
                          "derm_height": derm_height,
                          "derm_width": derm_width,
                          "derm_path": derm_path}
            else:
                sample = {"image_id": image_id,
                          "derm_height": derm_height,
                          "derm_width": derm_width,
                          "derm_path": derm_path}

            annos.append(sample)
        converted_annos = {"annotations": annos,
                           "num_classes": num_classes}
        save_path = os.path.join(args.sp, "converted_" + os.path.splitext(os.path.split(args.csv_file)[-1])[0] + ".json")
        print("Converted, Saving converted file to {}".format(save_path))
        with open(save_path, "w") as f:
            json.dump(converted_annos, f)


if __name__ == "__main__":
    args = parse_args()
    convert(args)

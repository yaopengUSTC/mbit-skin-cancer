import json, os
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd
import pickle


def parse_args():
    '''
    example: python convert_from_ISIC_2018.py
                --csv_file ./dataset_files/ISIC_2018_Train.csv
                --pkl_file ./dataset_files/ISIC_2018_5foldcv_indices.pkl
                --root /work/image/skin_cancer/HAM10000/training_set/HAM10000_images/
                --sp ../main/jsons/
    '''
    parser = argparse.ArgumentParser(description="Get the json file for ISIC 2018 dataset.")

    parser.add_argument(
        "--csv_file",
        help="origin csv file to be converted",
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
    num_classes = 7
    class_dict = {'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3, 'bkl': 4, 'df': 5, 'vasc': 6}
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
                    image_id = csv_data.iloc[i]['image_id']
                    label = int(class_dict[csv_data.iloc[i]["dx"]])
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
                                         "converted_" + "ISIC_2018_Train_" + str(j) + ".json")
                print("Converted, Saving converted file to {}".format(save_path))
                with open(save_path, "w") as f:
                    json.dump(converted_annos, f)

                new_annos = []
                for ii in tqdm(range(len(val_idx))):
                    i = val_idx[ii]
                    image_id = csv_data.iloc[i]['image_id']
                    label = int(class_dict[csv_data.iloc[i]["dx"]])
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
                                         "converted_" + "ISIC_2018_Val_" + str(j) + ".json")
                print("Converted, Saving converted file to {}".format(save_path))
                with open(save_path, "w") as f:
                    json.dump(converted_annos, f)
    else:   # all images in the dataset are used
        annos = []
        for i in tqdm(range(len(csv_data))):
            image_id = csv_data.iloc[i]['image_id']
            if "Train" in args.csv_file:    # for train
                label = int(class_dict[csv_data.iloc[i]["dx"]])
            elif "Val" in args.csv_file:    # for val
                label = int(csv_data.iloc[i]["MEL"]) \
                    + int(csv_data.iloc[i]["NV"]) * 2 \
                    + int(csv_data.iloc[i]["BCC"]) * 3 \
                    + int(csv_data.iloc[i]["AKIEC"]) * 4 \
                    + int(csv_data.iloc[i]["BKL"]) * 5 \
                    + int(csv_data.iloc[i]["DF"]) * 6 \
                    + int(csv_data.iloc[i]["VASC"]) * 7 \
                    - 1
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
        save_path = os.path.join(args.sp, "converted_" + os.path.splitext(os.path.split(args.file)[-1])[0] + ".json")
        print("Converted, Saveing converted file to {}".format(save_path))
        with open(save_path, "w") as f:
            json.dump(converted_annos, f)


if __name__ == "__main__":
    args = parse_args()
    convert(args)

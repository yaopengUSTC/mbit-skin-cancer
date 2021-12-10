import json, os
import argparse
from tqdm import tqdm
from PIL import Image


def parse_args():
    '''
    example: python convert_from_Derm_7pt.py
                --file ./dataset_files/Derm_7pt_train.json
                --root /work/image/skin_cancer/derm_7_point/images/
                --sp ../main/jsons/
    '''
    parser = argparse.ArgumentParser(description="Get the json file for Derm_7pt dataset.")

    parser.add_argument(
        "--file",
        help="origin json file to be converted",
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
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args


def convert(json_file, image_root):
    all_annos = json.load(open(json_file, 'r'))

    new_annos = []
    print("Converting file {} ...".format(json_file))
    max_class = 0
    for anno in tqdm(all_annos):
        clinic_path = os.path.join(image_root, anno["clinic_path"])
        assert os.path.exists(clinic_path), \
            'Image file does not exist: {}'.format(clinic_path)
        sizes = Image.open(clinic_path).size
        clinic_width = sizes[0]
        clinic_height = sizes[1]

        derm_path = os.path.join(image_root, anno["derm_path"])
        assert os.path.exists(derm_path), \
            'Image file does not exist: {}'.format(derm_path)
        sizes = Image.open(derm_path).size
        derm_width = sizes[0]
        derm_height = sizes[1]
        if max_class < anno["category_id"]:
            max_class = anno["category_id"]
        new_annos.append({"image_id": anno["image_id"],
                          "category_id": anno["category_id"],
                          "clinic_height": clinic_height,
                          "clinic_width": clinic_width,
                          "clinic_path": clinic_path,
                          "derm_height": derm_height,
                          "derm_width": derm_width,
                          "derm_path": derm_path})
    num_classes = max_class + 1
    return {"annotations": new_annos,
            "num_classes": num_classes}


if __name__ == "__main__":
    args = parse_args()
    converted_annos = convert(args.file, args.root)
    save_path = os.path.join(args.sp, "converted_" + os.path.split(args.file)[-1])
    print("Converted, Saveing converted file to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)



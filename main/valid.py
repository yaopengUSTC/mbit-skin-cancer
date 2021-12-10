import _init_paths
from lib.net import Network
from lib.config import cfg, update_config, update_cfg_name
from lib.dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from lib.core.evaluate import FusionMatrix
import torch.backends.cudnn as cudnn
from lib.core.function import get_val_result, get_test_result, get_roc_auc
import ast
import csv


def parse_args():
    '''
    example: python valid.py
                 --cfg ../configs/isic_2018.yaml
                 --test False
    :return:
    '''
    parser = argparse.ArgumentParser(description="codes for MBIT")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="",
        type=str,
    )
    parser.add_argument(
        "--test",
        help="decide whether to be test or valid",
        required=False,
        default=False,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def test_model(data_loader, model, cfg, device):
    model.eval()
    num_classes = data_loader.dataset.get_num_classes()
    if cfg.DATASET.VALID_ADD_ONE_CLASS:
        func = torch.nn.Sigmoid()
        num_classes -= 1
        func_1 = torch.nn.Softmax(dim=0)
    else:
        func = torch.nn.Softmax(dim=1)
        if cfg.DATASET.DATASET == "isic_2019":
            num_classes -= 1       # For isic_2019, the number of classes in the training set is 8, and there is also a class "unk" in the test set

    all_preds = np.zeros([data_loader.dataset.get_num_images(), num_classes])
    ii = 0
    print("\n-------  Start testing  -------")
    print("         Test json file: {} \n".format(cfg.DATASET.TEST_JSON))
    pbar = tqdm(total=len(data_loader))
    with torch.no_grad():
        for i, (image, label, meta) in enumerate(data_loader):
            image = image.to(device)
            if cfg.BACKBONE.BBN:        # BBN model
                feature = model(image, feature_flag=True)
                output = model(feature, classifier_flag=True)
            else:
                output = model(image)
            now_pred = func(output)
            new_pred = get_test_result(now_pred.cpu().numpy(), cfg, num_classes)
            all_preds[ii:ii + new_pred.shape[0], :] = new_pred
            ii += new_pred.shape[0]
            pbar.update(1)

    pbar.close()

    if cfg.DATASET.VALID_ADD_ONE_CLASS:
        all_unk = 1.0 - np.amax(all_preds, axis=1)
        all_preds = np.c_[all_preds, all_unk]
    elif cfg.DATASET.DATASET == "isic_2019":
        all_unk = np.zeros(all_preds.shape[0])  # For isic_2019, the results of the last class are all 0
        all_preds = np.c_[all_preds, all_unk]

    image_id_list = data_loader.dataset.get_image_id_list()

    # save the predictions to csv file
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    model_file = model_file.split('\\')[-1].split('.')[0]
    csv_file = "pridiction_" + str(model_file) + '.csv'
    csv_file = os.path.join(model_dir, csv_file)
    csv_fp = open(csv_file, 'w')
    csv_writer = csv.writer(csv_fp)
    table_head = ['image'] + cfg.DATASET.CLASS_NAME
    if cfg.DATASET.VALID_ADD_ONE_CLASS or cfg.DATASET.DATASET == "isic_2019":
        table_head = table_head + [cfg.DATASET.ADD_CLASS_NAME]
    # elif cfg.DATASET.DATASET == "isic_2017":
    #     csv_writer.writerow(['image', 'MEL', 'SK', 'NV'])
    # elif cfg.DATASET.DATASET == "isic_2018":
    #     csv_writer.writerow(['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
    # elif cfg.DATASET.DATASET == "isic_2019":
    #     csv_writer.writerow(['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])
    csv_writer.writerow(table_head)
    for i in range(len(image_id_list)):
        # change predictions to binary value([0, 1])
        pred = all_preds[i]
        if cfg.DATASET.VALID_ADD_ONE_CLASS:
            pred = torch.from_numpy(pred)
            pred = func_1(pred).numpy()
        pred = pred.tolist()
        # for every row, change the max prediction to 0.51 if the max prediction is lower than 0.5
        row_new = pred
        if max(pred) <= 0.5:
            row_new[pred.index(max(pred))] = 0.51
        row_new = [image_id_list[i]] + row_new
        csv_writer.writerow(row_new)

    # close the csv file
    csv_fp.close()


def valid_model(data_loader, model, cfg, device):
    model.eval()
    num_classes = data_loader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)
    if cfg.DATASET.VALID_ADD_ONE_CLASS:
        func = torch.nn.Sigmoid()
        num_classes -= 1
    else:
        func = torch.nn.Softmax(dim=1)

    all_preds = np.zeros([data_loader.dataset.get_num_images(), num_classes])
    all_labels = np.zeros(data_loader.dataset.get_num_images())
    ii = 0
    print("\n-------  Start validation  -------")
    print("         Valid json file: {} \n".format(cfg.DATASET.VALID_JSON))
    pbar = tqdm(total=len(data_loader))
    with torch.no_grad():
        for i, (image, label, meta) in enumerate(data_loader):
            image, label = image.to(device), label.to(device)
            if cfg.BACKBONE.BBN:  # BBN model
                feature = model(image, feature_flag=True)
                output = model(feature, classifier_flag=True)
            else:
                output = model(image)
            now_pred = func(output)
            new_pred, new_label = get_val_result(now_pred.cpu().numpy(), label.cpu().numpy(), cfg, num_classes)

            all_preds[ii:ii + new_pred.shape[0], :] = new_pred
            all_labels[ii:ii + new_pred.shape[0]] = new_label
            ii += new_pred.shape[0]
            pbar.update(1)

    pbar.close()

    if cfg.DATASET.VALID_ADD_ONE_CLASS:
        all_max = 1.0 - np.amax(all_preds, axis=1)
        all_preds = np.c_[all_preds, all_max]

    all_result = np.argmax(all_preds, 1)
    fusion_matrix.update(all_result, all_labels)
    roc_auc = get_roc_auc(all_preds, all_labels)

    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["specificity"] = fusion_matrix.get_pre_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = roc_auc
    metrics["fusion_matrix"] = fusion_matrix.matrix
    if cfg.DATASET.DATASET == "isic_2017":
        acc_1 = fusion_matrix.get_binary_accuracy(0)
        acc_2 = fusion_matrix.get_binary_accuracy(1)
        metrics["acc"] = (acc_1 + acc_2) / 2.0
        metrics["bacc"] = (metrics["sensitivity"][0] + metrics["sensitivity"][1]) / 2.0
        auc_mean = (metrics["roc_auc"][0] + metrics["roc_auc"][1]) / 2.0
        spec_mean = (metrics["specificity"][0] + metrics["specificity"][1]) / 2.0
    else:
        metrics["acc"] = fusion_matrix.get_accuracy()
        metrics["bacc"] = fusion_matrix.get_balance_accuracy()
        auc_mean = np.mean(metrics["roc_auc"])
        spec_mean = np.mean(metrics["specificity"])

    print("\n-------  Valid result: Valid_Acc: {:>6.3f}%  Balance_Acc: {:>6.3f}%  -------".format(
        metrics["acc"] * 100, metrics["bacc"] * 100)
    )
    print("         roc_auc.mean: {:>6.3f}  f1_score: {:>6.4f}     ".format(
        auc_mean, metrics["f1_score"])
    )
    print("         roc_auc:       {}  ".format(metrics["roc_auc"]))
    print("         sensitivity:   {}  ".format(metrics["sensitivity"]))
    print("         specificity:   {}   mean:   {}  ".format(metrics["specificity"], spec_mean))
    print("         fusion_matrix: \n{}  ".format(metrics["fusion_matrix"]))


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    update_cfg_name(cfg)  # modify the cfg.NAME

    cudnn.benchmark = True
    strGPUs = [str(x) for x in cfg.GPUS]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(strGPUs)  # specify which GPUs to use

    if args.test:
        valid_test_set = eval(cfg.DATASET.DATASET)("test", cfg)
    else:
        valid_test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = valid_test_set.get_num_classes()
    if cfg.DATASET.VALID_ADD_ONE_CLASS or (cfg.DATASET.DATASET == "isic_2019" and args.test):
        num_classes -= 1      # for isic_2019, the number of classes in the training set is 8, and there is also a class "unk" in the test set

    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    model = Network(cfg, mode="val", num_classes=num_classes)

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    print("have loaded the best model from {}".format(model_path))

    checkpoint = torch.load(
        model_path, map_location="cpu" if cfg.CPU_MODE else "cuda"
    )
    model.load_state_dict(checkpoint['state_dict'])

    test_batch_size = cfg.TEST.BATCH_SIZE * len(cfg.GPUS)
    sample_repeat_num = 0
    if cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE:
        sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM
    if cfg.TRAIN.SAMPLER.MULTI_SCALE.ENABLE:
        sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM
    if sample_repeat_num > 0:
        test_batch_size = int(test_batch_size / sample_repeat_num) * sample_repeat_num
    val_test_loader = DataLoader(
        valid_test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    if args.test:
        test_model(val_test_loader, model, cfg, device)
    else:
        valid_model(val_test_loader, model, cfg, device)



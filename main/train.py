import _init_paths
from lib.loss import *
from lib.dataset import *
from lib.config import cfg, update_config, update_cfg_name
from lib.utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
)
from lib.core.function import train_model, valid_model
from lib.core.combiner import Combiner

import numpy as np
import torch
import os, shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import ast
import csv


def parse_args():
    '''
    example: python train.py
                --cfg ../configs/isic_2018.yaml
                NAME "_1" LOSS.LOSS_TYPE "FocalLoss"
    :return:
    '''
    parser = argparse.ArgumentParser(description="codes for MBIT")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../configs/isic_2017.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type=ast.literal_eval,
        dest='auto_resume',
        required=False,
        default=True,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    update_cfg_name(cfg)  # modify the cfg.NAME
    logger, log_file = create_logger(cfg)
    warnings.filterwarnings("ignore")
    cudnn.benchmark = True
    auto_resume = args.auto_resume
    strGPUs = [str(x) for x in cfg.GPUS]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(strGPUs)  # specify which GPUs to use

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)

    num_classes = train_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    num_class_list = train_set.get_num_class_list()

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }

    # ----- begin model builder -----
    max_epoch = cfg.TRAIN.MAX_EPOCH
    loss_fuc = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
    model = get_model(cfg, num_classes, device, logger)
    combiner = Combiner(cfg, device, model, loss_fuc)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    # ----- end model builder -----

    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )

    test_batch_size = cfg.TEST.BATCH_SIZE * len(cfg.GPUS)
    val_sample_repeat_num = 0
    if cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE:
        val_sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM
    if cfg.TRAIN.SAMPLER.MULTI_SCALE.ENABLE:
        val_sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM
    if val_sample_repeat_num > 0:
        test_batch_size = int(test_batch_size / val_sample_repeat_num) * val_sample_repeat_num
    validLoader = DataLoader(
        valid_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard")
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        )
        if not click.confirm(
                "\033[1;31;40mContinue and override the former directory?\033[0m",
                default=False,
        ):
            exit(0)
        shutil.rmtree(code_dir)
        if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", ".pyth", "*build*", "*output*", "*datasets*",
        "pretrained_models"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    if tensorboard_dir is not None:
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_graph(model if cfg.CPU_MODE else model.module, (dummy_input,))
    else:
        writer = None

    best_result, best_epoch, start_epoch = 0, 0, 1
    best_metrics = {}

    # ----- begin resume ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume is False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg.CPU_MODE else "cuda"
        )

        model.load_state_dict(checkpoint['state_dict'])
        if cfg.RESUME_MODE != "state_dict":
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            best_epoch = checkpoint['best_epoch']
            best_metrics = checkpoint['best_metrics']
    # ----- end resume ---------

    logger.info("----------  Train start :  model: {},  loss: {},  need augment: {}  ----------".format(
        cfg.BACKBONE.TYPE + ("_BBN" if cfg.BACKBONE.BBN else ""),
        cfg.LOSS.LOSS_TYPE,
        str(cfg.TRAIN.SAMPLER.AUGMENT.NEED_AUGMENT))
    )

    for epoch in range(start_epoch, max_epoch + 1):
        scheduler.step()
        train_acc, train_loss = train_model(
            trainLoader, epoch, max_epoch, optimizer,
            combiner, cfg, logger, writer=writer
        )

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            val_metrics = valid_model(
                validLoader, epoch, combiner, cfg,
                logger, writer=writer
            )
            loss_dict["valid_loss"], acc_dict["valid_acc"] = val_metrics["loss"], val_metrics["acc"]

            compare_result = val_metrics["bacc"]  # for all dataset, take the BACC as the key metric
            if compare_result > best_result:
                best_result, best_epoch = compare_result, epoch
                best_metrics = val_metrics
                torch.save(
                    {'state_dict': model.state_dict(),
                     'epoch': epoch,
                     'best_result': best_result,
                     'best_epoch': best_epoch,
                     'best_metrics': best_metrics,
                     'scheduler': scheduler.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     }, os.path.join(model_dir, "best_model.pth")
                )
            logger.info(
                "--------------  Best_Epoch:{:>3d}    Best_result:{:>6.3f}%  -------------- \n".format(
                    best_epoch, best_result * 100
                )
            )

        model_save_path = os.path.join(model_dir, "epoch_{}.pth".format(epoch))
        if epoch % cfg.SAVE_STEP == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'best_metrics': best_metrics,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)

        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
    logger.info(
        "------------------- Train Finished :{} -------------------\n\n".format(cfg.NAME)
    )

    if cfg.DATASET.DATASET == "isic_2017":
        auc_mean = (best_metrics["roc_auc"][0] + best_metrics["roc_auc"][1]) / 2.0
        spec_mean = np.mean(best_metrics["specificity"])
    else:
        auc_mean = np.mean(best_metrics["roc_auc"])
        spec_mean = np.mean(best_metrics["specificity"])

    logger.info("-------  Best Valid: Epoch: {:>3d}  Valid_Loss: {:>6.4f}   Valid_Acc: {:>6.3f}%  -------".format(
        best_epoch, best_metrics["loss"], best_metrics["acc"] * 100)
    )
    logger.info("         roc_auc.mean: {:>6.3f}  f1_score: {:>6.4f}  Balance_Acc: {:>6.3f}%   ".format(
        auc_mean, best_metrics["f1_score"], best_metrics["bacc"] * 100)
    )
    logger.info("         roc_auc:       {}  ".format(best_metrics["roc_auc"]))
    logger.info("         sensitivity:   {}  ".format(best_metrics["sensitivity"]))
    logger.info("         specificity:   {}   mean:   {}  ".format(best_metrics["specificity"], spec_mean))
    logger.info("         fusion_matrix: \n{}\n  ".format(best_metrics["fusion_matrix"]))



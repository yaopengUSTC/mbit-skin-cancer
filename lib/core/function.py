from .evaluate import AverageMeter, FusionMatrix
import numpy as np
import torch
from torch.nn import functional as F
import time
from sklearn.metrics import auc, roc_curve


def train_model(
        train_loader, epoch, epoch_number, optimizer,
        combiner, cfg, logger, **kwargs
):
    if cfg.EVAL_MODE:
        combiner.model.eval()
    else:
        combiner.model.train()

    combiner.reset_epoch(epoch)

    start_time = time.time()
    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(train_loader):
        cnt = label.shape[0]
        loss, now_acc, _ = combiner.forward(image, label, meta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   l_rate:{:>9.7f}   Avg_Loss:{:>6.4f}   Epoch_Acc:{:>6.3f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, optimizer.param_groups[0]['lr'], all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model(
        val_loader, epoch, combiner, cfg, logger, **kwargs
):
    combiner.model.eval()

    num_classes = val_loader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)
    if cfg.DATASET.VALID_ADD_ONE_CLASS:
        num_classes = val_loader.dataset.get_num_classes() - 1

    all_preds = np.zeros([val_loader.dataset.get_num_images(), num_classes])
    all_labels = np.zeros(val_loader.dataset.get_num_images())
    ii = 0
    with torch.no_grad():
        all_loss = AverageMeter()
        for i, (image, label, meta) in enumerate(val_loader):
            cnt = label.shape[0]
            loss, now_acc, now_pred = combiner.forward(image, label, meta)
            now_pred = now_pred.cpu().numpy()
            new_pred, new_label = get_val_result(now_pred, label.cpu().numpy(), cfg, num_classes)

            all_preds[ii:ii + new_pred.shape[0], :] = new_pred
            all_labels[ii:ii + new_pred.shape[0]] = new_label
            ii += new_pred.shape[0]
            all_loss.update(loss.data.item(), cnt)

    if cfg.DATASET.VALID_ADD_ONE_CLASS:
        all_max = 1.0 - np.amax(all_preds, axis=1)
        all_preds = np.c_[all_preds, all_max]

    all_result = np.argmax(all_preds, 1)
    fusion_matrix.update(all_result, all_labels)
    roc_auc = get_roc_auc(all_preds, all_labels)

    metrics = {}
    metrics["loss"] = all_loss.avg
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["specificity"] = fusion_matrix.get_pre_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = roc_auc
    metrics["fusion_matrix"] = fusion_matrix.matrix
    if cfg.DATASET.DATASET == "isic_2017":
        auc_mean = (metrics["roc_auc"][0] + metrics["roc_auc"][1]) / 2.0
    else:
        auc_mean = np.mean(metrics["roc_auc"])
    metrics["acc"] = fusion_matrix.get_accuracy()
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    spec_mean = np.mean(metrics["specificity"])

    logger.info("-------  Valid: Epoch: {:>3d}  Valid_Loss: {:>6.4f}   Valid_Acc: {:>6.3f}%  -------".format(
        epoch, metrics["loss"], metrics["acc"] * 100)
    )
    logger.info("         roc_auc.mean: {:>6.3f}  f1_score: {:>6.4f}  Balance_Acc: {:>6.3f}%   ".format(
        auc_mean, metrics["f1_score"], metrics["bacc"] * 100)
    )
    logger.info("         roc_auc:       {}  ".format(metrics["roc_auc"]))
    logger.info("         sensitivity:   {}  ".format(metrics["sensitivity"]))
    logger.info("         specificity:   {}   mean:   {}  ".format(metrics["specificity"], spec_mean))
    logger.info("         fusion_matrix: \n{}  ".format(metrics["fusion_matrix"]))

    return metrics


def get_val_result(pred, label, cfg, num_classes):
    val_sample_repeat_num = 0
    if cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE:
        val_sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM
    if cfg.TRAIN.SAMPLER.MULTI_SCALE.ENABLE:
        val_sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM

    if val_sample_repeat_num > 0:
        pred = pred.reshape(-1, val_sample_repeat_num, num_classes)
        pred = np.transpose(pred, (0, 2, 1))
        label = label.reshape(-1, val_sample_repeat_num)
        label = label[:, 0]

        if cfg.TRAIN.SAMPLER.MULTI_CROP.SCHEME == 'vote':
            pred = np.argmax(pred, 1)
            predictions = np.zeros([pred.shape[0], num_classes])
            for j in range(pred.shape[0]):
                predictions[j, :] = np.bincount(pred[j, :], minlength=num_classes)
                predictions[j, :] = predictions[j, :] / np.sum(predictions[j, :])  # normalize
        else:       # 'average'
            predictions = np.mean(pred, 2)
    else:
        predictions = pred

    return predictions, label


def get_test_result(pred, cfg, num_classes):
    val_sample_repeat_num = 0
    if cfg.TRAIN.SAMPLER.MULTI_CROP.ENABLE:
        val_sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_CROP.CROP_NUM
    if cfg.TRAIN.SAMPLER.MULTI_SCALE.ENABLE:
        val_sample_repeat_num += cfg.TRAIN.SAMPLER.MULTI_SCALE.SCALE_NUM

    if val_sample_repeat_num > 0:
        pred = pred.reshape(-1, val_sample_repeat_num, num_classes)
        pred = np.transpose(pred, (0, 2, 1))

        if cfg.TRAIN.SAMPLER.MULTI_CROP.SCHEME == 'vote':
            pred = np.argmax(pred, 1)
            predictions = np.zeros([pred.shape[0], num_classes])
            for j in range(pred.shape[0]):
                predictions[j, :] = np.bincount(pred[j, :], minlength=num_classes)
                predictions[j, :] = predictions[j, :] / np.sum(predictions[j, :])  # normalize
        else:       # 'average'
            predictions = np.mean(pred, 2)
    else:
        predictions = pred

    return predictions


def get_roc_auc(all_preds, all_labels):
    one_hot = label_to_one_hot(all_labels, all_preds.shape[1])

    fpr = {}
    tpr = {}
    roc_auc = np.zeros([all_preds.shape[1]])
    for i in range(all_preds.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(one_hot[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc


def label_to_one_hot(label, num_class):
    one_hot = F.one_hot(torch.from_numpy(label).long(), num_class).float()
    one_hot = one_hot.numpy()

    return one_hot


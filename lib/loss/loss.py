import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from cvxopt import matrix, spdiag, solvers


class BaseLoss(nn.Module):
    def __init__(self, para_dict=None):
        super(BaseLoss, self).__init__()
        self.num_class_list = np.array(para_dict["num_class_list"])
        self.no_of_class = len(self.num_class_list)
        self.device = para_dict["device"]

        self.cfg = para_dict["cfg"]
        self.class_weight_power = self.cfg.LOSS.WEIGHT_POWER
        self.class_extra_weight = np.array(self.cfg.LOSS.EXTRA_WEIGHT)
        self.scheduler = self.cfg.LOSS.SCHEDULER
        self.drw_epoch = self.cfg.LOSS.DRW_EPOCH
        self.cls_epoch_min = self.cfg.LOSS.CLS_EPOCH_MIN
        self.cls_epoch_max = self.cfg.LOSS.CLS_EPOCH_MAX
        self.weight = None

    def reset_epoch(self, epoch):
        if self.scheduler == "default":     # the weights of all classes are "1.0"
            per_cls_weights = np.array([1.0] * self.no_of_class)
        elif self.scheduler == "re_weight":
            per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
            per_cls_weights = per_cls_weights * self.class_extra_weight
            per_cls_weights = [math.pow(num, self.class_weight_power) for num in per_cls_weights]
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        elif self.scheduler == "drw":       # two-stage strategy using re-weighting at the second stage
            if epoch < self.drw_epoch:
                per_cls_weights = np.array([1.0] * self.no_of_class)
            else:
                per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
                per_cls_weights = per_cls_weights * self.class_extra_weight
                per_cls_weights = [math.pow(num, self.class_weight_power) for num in per_cls_weights]
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        elif self.scheduler == "cls":       # cumulative learning strategy
            if epoch <= self.cls_epoch_min:
                now_power = 0
            elif epoch < self.cls_epoch_max:
                now_power = ((epoch - self.cls_epoch_min) / (self.cls_epoch_max - self.cls_epoch_min)) ** 2
                now_power = now_power * self.class_weight_power
            else:
                now_power = self.class_weight_power

            per_cls_weights = 1.0 / (self.num_class_list.astype(np.float64))
            per_cls_weights = per_cls_weights * self.class_extra_weight
            per_cls_weights = [math.pow(num, now_power) for num in per_cls_weights]
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * self.no_of_class
        else:
            raise AttributeError(
                "loss scheduler can only be 'default', 're_weight', 'drw' and 'cls'.")

        print("class weight of loss: {}".format(per_cls_weights))
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)


class CrossEntropy(BaseLoss):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__(para_dict)

    def forward(self, x, target):
        output = x
        loss = F.cross_entropy(output, target, weight=self.weight)
        return loss


class LDAMLoss(BaseLoss):
    """
    LDAM loss, Details of the theorem can be viewed in the paper
       "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"
        Copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).

    Args:
        max_m (float): the hyper-parameter of LDAM loss
    """
    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__(para_dict)

        max_m = self.cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        self.s = 30

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class FocalLoss(BaseLoss):
    """
    Focal loss, Details of the theorem can be viewed in the paper
       "Focal Loss for Dense Object Detection"

    Args:
        gamma (float): the hyper-parameter of focal loss
        type: "sigmoid", "cross_entropy" or "ldam"
    """
    def __init__(self, para_dict=None):
        super(FocalLoss, self).__init__(para_dict)

        self.gamma = self.cfg.LOSS.FOCAL.GAMMA
        self.type = self.cfg.LOSS.FOCAL.TYPE
        self.sigmoid = self.cfg.LOSS.FOCAL.SIGMOID
        if self.type == "ldam":
            max_m = self.cfg.LOSS.LDAM.MAX_MARGIN
            m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.FloatTensor(m_list).to(self.device)
            self.m_list = m_list
            self.s = 30

    def forward(self, x, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)

        if self.type == "sigmoid":
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, self.no_of_class)

            loss = F.binary_cross_entropy_with_logits(input=x, target=labels_one_hot, reduction="none")
            if self.gamma == 0.0:
                modulator = 1.0
            else:
                modulator = torch.exp(-self.gamma * labels_one_hot * x
                                      - self.gamma * torch.log(1 + torch.exp(-1.0 * x)))

            loss = modulator * loss
            weighted_loss = weights * loss
            if self.sigmoid == "enlarge":
                weighted_loss = torch.mean(weighted_loss) * 30
            else:
                weighted_loss = weighted_loss.sum() / weights.sum()
        elif self.type == "cross_entropy":
            loss = F.cross_entropy(x, target, reduction='none')

            p = torch.exp(-loss)
            loss = (1 - p) ** self.gamma * loss
            weighted_loss = weights * loss
            weighted_loss = weighted_loss.sum() / weights.sum()
        elif self.type == "ldam":
            index = torch.zeros_like(x, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)

            index_float = index.type(torch.FloatTensor)
            index_float = index_float.to(self.device)
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m

            output = torch.where(index, x_m, x)
            loss = F.cross_entropy(self.s * output, target, reduction='none')

            p = torch.exp(-loss)
            loss = (1 - p) ** self.gamma * loss
            weighted_loss = weights * loss
            weighted_loss = weighted_loss.sum() / weights.sum()
        else:
            raise AttributeError(
                "focal loss type can only be 'sigmoid', 'cross_entropy' and 'ldam'.")

        return weighted_loss


class LOWLoss(BaseLoss):
    """
    LOW loss, Details of the theorem can be viewed in the paper
       "LOW: Training Deep Neural Networks by Learning Optimal Sample Weights"

    Args:
        lamb (float): higher lamb means more smoothness -> weights closer to 1
    """
    def __init__(self, para_dict=None):
        super(LOWLoss, self).__init__(para_dict)

        self.lamb = self.cfg.LOSS.LOW.LAMB        #
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')  # can replace this with any loss with "reduction='none'"

    def forward(self, x, target):
        if x.requires_grad:         # for train
            # Compute loss gradient norm
            output_d = x.detach()
            loss_d = torch.mean(self.loss_func(output_d.requires_grad_(True), target), dim=0)
            loss_d.backward(torch.ones_like(loss_d))
            loss_grad = torch.norm(output_d.grad, 2, 1)

            # Computed weighted loss
            low_weights = self.compute_weights(loss_grad, self.lamb)
            loss = self.loss_func(x, target)
            low_loss = loss * low_weights
        else:                       # for valid
            low_loss = self.loss_func(x, target)

        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)

        weighted_loss = weights * low_loss
        weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss

    def compute_weights(self, loss_grad, lamb):
        device = loss_grad.get_device()
        loss_grad = loss_grad.data.cpu().numpy()

        # Compute Optimal sample Weights
        aux = -(loss_grad ** 2 + lamb)
        sz = len(loss_grad)
        P = 2 * matrix(lamb * np.identity(sz))
        q = matrix(aux.astype(np.double))
        A = spdiag(matrix(-1.0, (1, sz)))
        b = matrix(0.0, (sz, 1))
        Aeq = matrix(1.0, (1, sz))
        beq = matrix(1.0 * sz)
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 20
        solvers.options['abstol'] = 1e-4
        solvers.options['reltol'] = 1e-4
        solvers.options['feastol'] = 1e-4
        sol = solvers.qp(P, q, A, b, Aeq, beq)
        w = np.array(sol['x'])

        return torch.squeeze(torch.tensor(w, dtype=torch.float, device=device))


class GHMCLoss(BaseLoss):
    """
    GHM-C loss, Details of the theorem can be viewed in the paper
       "Gradient Harmonized Single-stage Detector".
       https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
    """
    def __init__(self, para_dict=None):
        super(GHMCLoss, self).__init__(para_dict)

        self.bins = self.cfg.LOSS.GHMC.BINS
        self.momentum = self.cfg.LOSS.GHMC.MOMENTUM
        self.edges = torch.arange(self.bins + 1).float().to(self.device) / self.bins
        self.edges[-1] += 1e-6
        if self.momentum > 0:
            self.acc_sum = torch.zeros(self.bins).to(self.device)

    def forward(self, pred, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_class)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - labels_one_hot)

        tot = pred.numel()
        ghm_weights = torch.zeros_like(pred)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] \
                        + (1 - self.momentum) * num_in_bin
                    ghm_weights[inds] = tot / self.acc_sum[i]
                else:
                    ghm_weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            ghm_weights = ghm_weights / n

        loss = F.binary_cross_entropy_with_logits(pred, labels_one_hot, reduction="none")
        ghm_loss = loss * ghm_weights

        weighted_loss = weights * ghm_loss
        weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss


class CCELoss(BaseLoss):
    """
    CCE loss, Details of the theorem can be viewed in the papers
       "Imbalanced Image Classification with Complement Cross Entropy" and
       "Complement objective training"
       https://github.com/henry8527/COT
    """
    def __init__(self, para_dict=None):
        super(CCELoss, self).__init__(para_dict)

    def forward(self, pred, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        labels_zero_hot = 1 - labels_one_hot
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)

        pred = F.softmax(pred, dim=1)
        y_g = torch.gather(pred, 1, torch.unsqueeze(target, 1))
        y_g_no = (1 - y_g) + 1e-7           # avoiding numerical issues (first)
        p_x = pred / y_g_no.view(len(pred), 1)
        p_x_log = torch.log(p_x + 1e-10)    # avoiding numerical issues (second)
        cce_loss = p_x * p_x_log * labels_zero_hot
        cce_loss = cce_loss.sum(1)
        cce_loss = cce_loss / (self.no_of_class - 1)

        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = (ce_loss + cce_loss) * weights
        weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss


class MWNLoss(BaseLoss):
    """
    Multi Weighted New loss
    Args:
        gamma (float): the hyper-parameter of focal loss
        beta (float, 0.0 - 0.4):
        type: "zero", "fix", "decrease"
        sigmoid: "normal", "enlarge"
    """
    def __init__(self, para_dict=None):
        super(MWNLoss, self).__init__(para_dict)

        self.gamma = self.cfg.LOSS.MWNL.GAMMA
        self.beta = self.cfg.LOSS.MWNL.BETA
        self.type = self.cfg.LOSS.MWNL.TYPE
        self.sigmoid = self.cfg.LOSS.MWNL.SIGMOID
        if self.beta > 0.4 or self.beta < 0.0:
            raise AttributeError(
                "For MWNLoss, the value of beta must be between 0.0 and 0.0 .")

    def forward(self, x, target):
        labels_one_hot = F.one_hot(target, self.no_of_class).float().to(self.device)
        weights = self.weight
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_class)

        loss = F.binary_cross_entropy_with_logits(input=x, target=labels_one_hot, reduction="none")

        if self.beta > 0.0:
            th = - math.log(self.beta)
            if self.type == "zero":
                other = torch.zeros(loss.shape).to(self.device)
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "fix":
                other = torch.ones(loss.shape).to(self.device)
                other = other * th
                loss = torch.where(loss <= th, loss, other)
            elif self.type == "decrease":
                pt = torch.exp(-1.0 * loss)
                loss = torch.where(loss <= th, loss, pt * th / self.beta)
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * x
                                  - self.gamma * torch.log(1 + torch.exp(-1.0 * x)))

        loss = modulator * loss

        weighted_loss = weights * loss
        if self.sigmoid == "enlarge":
            weighted_loss = torch.mean(weighted_loss) * 30
        else:
            weighted_loss = weighted_loss.sum() / weights.sum()
        return weighted_loss







import torch
from .evaluate import accuracy


class Combiner:
    def __init__(self, cfg, device, model, loss_func):
        self.cfg = cfg
        if cfg.BACKBONE.BBN:
            self.type = "bbn_mix"
        else:
            self.type = "default"
        self.device = device
        self.epoch = 0
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.model = model
        if cfg.DATASET.VALID_ADD_ONE_CLASS:
            self.func = torch.nn.Sigmoid()
        else:
            self.func = torch.nn.Softmax(dim=1)
        self.loss_func = loss_func
        self.alpha = 0.2
        self.div_epoch = self.max_epoch
        self.pre_freeze = self.cfg.BACKBONE.PRE_FREEZE

    def reset_epoch(self, epoch):
        self.epoch = epoch
        self.loss_func.reset_epoch(epoch)

        # Freeze the update of the model parameters first,
        # and then release it after a certain epoch.
        if self.pre_freeze and epoch > self.cfg.BACKBONE.PRE_FREEZE_EPOCH:
            self.pre_freeze = False
            if self.device == torch.device("cpu"):
                self.model.unfreeze_backbone()
            else:
                self.model.module.unfreeze_backbone()

    def forward(self, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            image, label, meta, **kwargs
        )

    def default(self, image, label, meta, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        output = self.model(image)
        if self.model.training or not self.cfg.DATASET.VALID_ADD_ONE_CLASS:
            loss = self.loss_func(output, label)
        else:
            loss = torch.zeros(1)       # for isic_2019 valid and test, ignore "loss"
        now_pred = self.func(output)
        now_result = torch.argmax(now_pred, 1)
        now_acc, _ = accuracy(now_result.cpu().numpy(), label.cpu().numpy())

        return loss, now_acc, now_pred

    # BBN method, please see the paper: "BBN: Bilateral-Branch Network with
    # Cumulative Learning for Long-Tailed Visual Recognition"
    def bbn_mix(self, image, label, meta, **kwargs):
        if self.model.training:
            image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
            label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

            feature_a, feature_b = (
                self.model(image_a, feature_cb=True),
                self.model(image_b, feature_rb=True),
            )

            if self.epoch < self.div_epoch:
                l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
            else:
                l = 0.0

            mixed_feature = 2 * torch.cat((l * feature_a, (1 - l) * feature_b), dim=1)
            output = self.model(mixed_feature, classifier_flag=True)
            loss = l * self.loss_func(output, label_a) + (1 - l) * self.loss_func(output, label_b)
            now_pred = self.func(output)
            now_result = torch.argmax(now_pred, 1)
            now_acc = (
                    l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                    + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
            )
        else:
            image, label = image.to(self.device), label.to(self.device)
            feature = self.model(image, feature_flag=True)
            output = self.model(feature, classifier_flag=True)
            if not self.cfg.DATASET.VALID_ADD_ONE_CLASS:
                loss = self.loss_func(output, label)
            else:
                loss = torch.zeros(1)  # ignore "loss"
            now_pred = self.func(output)
            now_result = torch.argmax(now_pred, 1)
            now_acc, _ = accuracy(now_result.cpu().numpy(), label.cpu().numpy())

        return loss, now_acc, now_pred

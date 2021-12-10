import torch
import torch.nn as nn
from lib.modules import GAP, Identity, FCNorm
import lib.backbone.all_models as all_models


class Network(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000):
        super(Network, self).__init__()
        pretrain = (
            True if mode == "train" and cfg.RESUME_MODEL == "" and cfg.BACKBONE.PRETRAINED
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg

        model = all_models.get_model(self.cfg.BACKBONE.TYPE, pretrain)
        # BBN method, please see the paper: "BBN: Bilateral-Branch Network with
        # Cumulative Learning for Long-Tailed Visual Recognition"
        if self.cfg.BACKBONE.BBN:
            self.model = all_models.get_bbn_model(self.cfg.BACKBONE.TYPE, model)
            self.bbn_backbone = self.model.bbn_backbone
            self.bbn_cb_block = self.model.bbn_cb_block
            self.bbn_rb_block = self.model.bbn_rb_block

            self.module = self._get_module()
            self.classifier = self._get_classifer()
        else:
            self.model, self.last_layer = all_models.modify_last_layer(self.cfg.BACKBONE.TYPE, model, self.num_classes,
                                                                       self.cfg)
        # if use drop block, set the parameter
        if 'DropBlock' in self.cfg.BACKBONE.TYPE:
            self.model.init_dropblock_para(start=0.0, stop=self.cfg.BACKBONE.DROP.BLOCK_PROB,
                                           block_size=self.cfg.BACKBONE.DROP.BLOCK_SIZE,
                                           nr_steps=self.cfg.BACKBONE.DROP.NR_STEPS)
            self.model.init_dropout_para(drop_prob=self.cfg.BACKBONE.DROP.OUT_PROB)

    def forward(self, x, **kwargs):
        if self.cfg.BACKBONE.BBN:  # BBN model
            if "feature_cb" in kwargs:              # used for train
                x = self.bbn_backbone(x)
                x = self.bbn_cb_block(x)
                x = self.module(x)
                x = x.view(x.shape[0], -1)
            elif "feature_rb" in kwargs:            # used for train
                x = self.bbn_backbone(x)
                x = self.bbn_rb_block(x)
                x = self.module(x)
                x = x.view(x.shape[0], -1)
            elif "feature_flag" in kwargs:          # used for valid
                x = self.bbn_backbone(x)
                out1 = self.bbn_cb_block(x)
                out2 = self.bbn_rb_block(x)
                out = torch.cat((out1, out2), dim=1)
                x = self.module(out)
                x = x.view(x.shape[0], -1)
            elif "classifier_flag" in kwargs:       # used for train and valid
                x = self.classifier(x)
                if 'DropBlock' in self.cfg.BACKBONE.TYPE:   # update drop block parameter
                    self.model.dropblock_step()
        else:   # Normal model
            x = self.model(x)
            if 'DropBlock' in self.cfg.BACKBONE.TYPE:       # update drop block parameter
                self.model.dropblock_step()

        return x

    def freeze_backbone(self):
        print("Freezing backbone .......")
        if self.cfg.BACKBONE.BBN:
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            for p in self.model.parameters():
                p.requires_grad = False
            # let the last classification layer parameter not freeze
            for p in self.last_layer.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self):
        print("Unfreezing backbone .......")
        for p in self.model.parameters():
            p.requires_grad = True

    def get_feature_length(self):
        num_features = all_models.get_feature_length(self.cfg.BACKBONE.TYPE, self.model)
        if self.cfg.BACKBONE.BBN:
            num_features *= 2
        return num_features

    def _get_module(self):
        """only for BBN model"""
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module = Identity()
        else:
            raise NotImplementedError

        return module

    def _get_classifer(self):
        """only for BBN model"""
        bias_flag = self.cfg.CLASSIFIER.BIAS

        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        else:
            raise NotImplementedError

        if 'DropBlock' in self.cfg.BACKBONE.TYPE:
            classifier = nn.Sequential(
                self.model.bbn_drop_out,
                classifier
            )

        return classifier

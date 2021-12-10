import torch.nn as nn
from lib.modules import FCNorm
import copy
from torchvision import models
import pretrainedmodels
from .efficientnet import efficientnet_b0
from .efficientnet import efficientnet_b1
from .efficientnet import efficientnet_b2
from .efficientnet import efficientnet_b3
from .efficientnet import efficientnet_b4
from .efficientnet import efficientnet_b5
from .efficientnet import efficientnet_b6
from .efficientnet import efficientnet_b7
from .build_regnet import RegNetY_200MF
from .build_regnet import RegNetX_200MF
from .build_regnet import RegNetY_400MF
from .build_regnet import RegNetX_400MF
from .build_regnet import RegNetY_600MF
from .build_regnet import RegNetX_600MF
from .build_regnet import RegNetY_800MF
from .build_regnet import RegNetX_800MF
from .build_regnet import RegNetY_1_6_GF
from .build_regnet import RegNetX_1_6_GF
from .build_regnet import RegNetX_3_2_GF
from .build_regnet import RegNetY_3_2_GF
from .build_regnet import RegNetY_4_0_GF
from .build_regnet import RegNetX_4_0_GF
from .build_regnet import RegNetX_6_4_GF
from .build_regnet import RegNetY_6_4_GF
from .build_regnet import RegNetY_8_0_GF
from .build_regnet import RegNetX_8_0_GF
from .build_regnet import RegNetX_1_2_0_GF
from .build_regnet import RegNetY_1_2_0_GF
from .build_regnet import RegNetY_1_6_0_GF
from .build_regnet import RegNetX_1_6_0_GF
from .build_regnet import RegNetY_3_2_0_GF
from .build_regnet import RegNetX_3_2_0_GF
from .build_regnet_dropblock import RegNetY_8_0_0_MF_DropBlock
from .build_regnet_dropblock import RegNetY_1_6_GF_DropBlock
from .build_regnet_dropblock import RegNetY_3_2_GF_DropBlock
from .build_regnet_dropblock import RegNetY_8_0_GF_DropBlock
from .build_regnet_dropblock import RegNetY_1_6_0_GF_DropBlock


def get_model(model_name, pretrained=False):
    """Returns a CNN model
    Args:
      model_name: model name
      pretrained: True or False
    Returns:
      model: the desired model
    Raises:
      ValueError: If model name is not recognized.
    """
    if pretrained == False:
        pt = None
    else:
        pt = 'imagenet'

    if model_name == 'Vgg11':
        return models.vgg11(pretrained=pretrained)
    elif model_name == 'Vgg13':
        return models.vgg13(pretrained=pretrained)
    elif model_name == 'Vgg16':
        return models.vgg16(pretrained=pretrained)
    elif model_name == 'Vgg19':
        return models.vgg19(pretrained=pretrained)
    elif model_name == 'Resnet18':
        return models.resnet18(pretrained=pretrained)
    elif model_name == 'Resnet34':
        return models.resnet34(pretrained=pretrained)
    elif model_name == 'Resnet50':
        return models.resnet50(pretrained=pretrained)
    elif model_name == 'Resnet101':
        return models.resnet101(pretrained=pretrained)
    elif model_name == 'Resnet152':
        return models.resnet152(pretrained=pretrained)
    elif model_name == 'Dense121':
        return models.densenet121(pretrained=pretrained)
    elif model_name == 'Dense169':
        return models.densenet169(pretrained=pretrained)
    elif model_name == 'Dense201':
        return models.densenet201(pretrained=pretrained)
    elif model_name == 'Dense161':
        return models.densenet161(pretrained=pretrained)
    elif model_name == 'SENet50':
        return pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=pt)
    elif model_name == 'SENet101':
        return pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained=pt)
    elif model_name == 'SENet152':
        return pretrainedmodels.__dict__['se_resnet152'](num_classes=1000, pretrained=pt)
    elif model_name == 'SENet154':
        return pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b0':
        return efficientnet_b0(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b1':
        return efficientnet_b1(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b2':
        return efficientnet_b2(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b3':
        return efficientnet_b3(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b4':
        return efficientnet_b4(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b5':
        return efficientnet_b5(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b6':
        return efficientnet_b6(num_classes=1000, pretrained=pt)
    elif model_name == 'Efficient_b7':
        return efficientnet_b7(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_200MF':
        return RegNetX_200MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_400MF':
        return RegNetX_400MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_600MF':
        return RegNetX_600MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_800MF':
        return RegNetX_800MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_1.6GF':
        return RegNetX_1_6_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_3.2GF':
        return RegNetX_3_2_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_4.0GF':
        return RegNetX_4_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_6.4GF':
        return RegNetX_6_4_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_8.0GF':
        return RegNetX_8_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_12GF':
        return RegNetX_1_2_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_16GF':
        return RegNetX_1_6_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetX_32GF':
        return RegNetX_3_2_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_200MF':
        return RegNetY_200MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_400MF':
        return RegNetY_400MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_600MF':
        return RegNetY_600MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_800MF':
        return RegNetY_800MF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_1.6GF':
        return RegNetY_1_6_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_3.2GF':
        return RegNetY_3_2_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_4.0GF':
        return RegNetY_4_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_6.4GF':
        return RegNetY_6_4_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_8.0GF':
        return RegNetY_8_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_12GF':
        return RegNetY_1_2_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_16GF':
        return RegNetY_1_6_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_32GF':
        return RegNetY_3_2_0_GF(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_800MF_DropBlock':
        return RegNetY_8_0_0_MF_DropBlock(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_1.6GF_DropBlock':
        return RegNetY_1_6_GF_DropBlock(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_3.2GF_DropBlock':
        return RegNetY_3_2_GF_DropBlock(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_8.0GF_DropBlock':
        return RegNetY_8_0_GF_DropBlock(num_classes=1000, pretrained=pt)
    elif model_name == 'RegNetY_16GF_DropBlock':
        return RegNetY_1_6_0_GF_DropBlock(num_classes=1000, pretrained=pt)
    else:
        raise ValueError('Name of model unknown %s' % model_name)


def modify_last_layer(model_name, model, num_classes, cfg):
    """modify the last layer of the CNN model to fit the num_classes
    Args:
      model_name: model name
      model: CNN model
      num_classes: class number
    Returns:
      model: the desired model
    """
    if 'Vgg' in model_name:
        num_ftrs = model.classifier._modules['6'].in_features
        model.classifier._modules['6'] = classifier(num_ftrs, num_classes, cfg)
        last_layer = model.classifier._modules['6']
    elif 'Dense' in model_name:
        num_ftrs = model.classifier.in_features
        model.classifier = classifier(num_ftrs, num_classes, cfg)
        last_layer = model.classifier
    elif 'Resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = classifier(num_ftrs, num_classes, cfg)
        last_layer = model.fc
    elif 'Efficient' in model_name:
        num_ftrs = model._fc.in_features
        model._fc = classifier(num_ftrs, num_classes, cfg)
        last_layer = model._fc
    elif 'RegNet' in model_name:
        num_ftrs = model.head.fc.in_features
        model.head.fc = classifier(num_ftrs, num_classes, cfg)
        last_layer = model.head.fc
    else:
        num_ftrs = model.last_linear.in_features
        model.last_linear = classifier(num_ftrs, num_classes, cfg)
        last_layer = model.last_linear
    # print(model)
    return model, last_layer


def classifier(num_features, num_classes, cfg):
    bias_flag = cfg.CLASSIFIER.BIAS

    if cfg.CLASSIFIER.TYPE == "FCNorm":
        last_linear = FCNorm(num_features, num_classes)
    elif cfg.CLASSIFIER.TYPE == "FC":
        last_linear = nn.Linear(num_features, num_classes, bias=bias_flag)
    else:
        raise NotImplementedError
    return last_linear


def get_bbn_model(model_name, model):
    """modify the normal CNN model to BBN model. Only supports Resnet and RegNet model.
    Args:
      model_name: model name
      model: CNN model
    Returns:
      model: the desired model
    """
    if 'Resnet' in model_name:
        model.bbn_backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3
        )
        model.bbn_cb_block = model.layer4
        model.bbn_rb_block = copy.deepcopy(model.bbn_cb_block)
    elif 'RegNet' in model_name:
        if 'DropBlock' in model_name:
            model.bbn_backbone = nn.Sequential(
                model.stem,
                model.s1,
                model.s2,
                model.s3,
                model.s3_drop_block
            )
            model.bbn_cb_block = nn.Sequential(
                model.s4,
                model.s4_drop_block
            )
            model.bbn_drop_out = model.head.dropout
        else:
            model.bbn_backbone = nn.Sequential(
                model.stem,
                model.s1,
                model.s2,
                model.s3
            )
            model.bbn_cb_block = model.s4

        model.bbn_rb_block = copy.deepcopy(model.bbn_cb_block)
    else:
        raise ValueError('Model %s does not support bbn structure.' % model_name)

    return model


def get_feature_length(model_name, model):
    """get the feature length of the last feature layer
    Args:
      model_name: model name
      model: CNN model
    Returns:
      num_ftrs: the feature length of the last feature layer
    """
    if 'Vgg' in model_name:
        num_ftrs = model.classifier._modules['6'].in_features
    elif 'Dense' in model_name:
        num_ftrs = model.classifier.in_features
    elif 'Resnet' in model_name:
        num_ftrs = model.fc.in_features
    elif 'Efficient' in model_name:
        num_ftrs = model._fc.in_features
    elif 'RegNet' in model_name:
        num_ftrs = model.head.fc.in_features
    else:
        num_ftrs = model.last_linear.in_features

    return num_ftrs


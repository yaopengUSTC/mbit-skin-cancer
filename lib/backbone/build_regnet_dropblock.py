from .regnet_dropblock import RegNet_DropBlock
import torch
import os
import yaml


pretrained_settings = {
    'regnet': {
        'imagenet': {
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], 'num_classes should be {}, but is {}'.format(
        settings['num_classes'], num_classes)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def RegNetY_8_0_0_MF_DropBlock(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-800MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet_DropBlock(config)
    # print(model)
    last_checkpoint = '../pretrained_models/RegNetY-800MF_dds_8gpu.pyth'
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
    checkpoint = torch.load(last_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print('have loaded checkpoint from {}'.format(last_checkpoint))
    if pretrained is not None:
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_1_6_GF_DropBlock(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-1.6GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet_DropBlock(config)
    # print(model)
    last_checkpoint = '../pretrained_models/RegNetY-1.6GF_dds_8gpu.pyth'
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
    checkpoint = torch.load(last_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print('have loaded checkpoint from {}'.format(last_checkpoint))
    if pretrained is not None:
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_3_2_GF_DropBlock(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-3.2GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet_DropBlock(config)
    # print(model)
    last_checkpoint = '../pretrained_models/RegNetY-3.2GF_dds_8gpu.pyth'
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
    checkpoint = torch.load(last_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print('have loaded checkpoint from {}'.format(last_checkpoint))
    if pretrained is not None:
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_8_0_GF_DropBlock(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-8.0GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet_DropBlock(config)
    # print(model)
    last_checkpoint = '../pretrained_models/RegNetY-8.0GF_dds_8gpu.pyth'
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
    checkpoint = torch.load(last_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print('have loaded checkpoint from {}'.format(last_checkpoint))
    if pretrained is not None:
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_1_6_0_GF_DropBlock(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-16GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet_DropBlock(config)
    # print(model)
    last_checkpoint = '../pretrained_models/RegNetY-16GF_dds_8gpu.pyth'
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
    checkpoint = torch.load(last_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print('have loaded checkpoint from {}'.format(last_checkpoint))
    if pretrained is not None:
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


if __name__ == "__main__":
    RegNetY_3_2_GF_DropBlock()
    print('success')


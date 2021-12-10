from .regnet import RegNet
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


def RegNetY_200MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-200MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-200MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_200MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-200MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-200MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_400MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-400MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-400MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_400MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-400MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-400MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_600MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-600MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-600MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_600MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-600MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-600MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_800MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-800MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-800MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_800MF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-800MF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-800MF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_1_6_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-1.6GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-1.6GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_1_6_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-1.6GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-1.6GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_3_2_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-3.2GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-3.2GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_3_2_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-3.2GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-3.2GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_4_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-4.0GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-4.0GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_4_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-4.0GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-4.0GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_6_4_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-6.4GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-6.4GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_6_4_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-6.4GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-6.4GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_8_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-8.0GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-8.0GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_8_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-8.0GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-8.0GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_1_2_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-12GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-12GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_1_2_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-12GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-12GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_1_6_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-16GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-16GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_1_6_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-16GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-16GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetY_3_2_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetY-32GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetY-32GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def RegNetX_3_2_0_GF(num_classes=1000, pretrained='imagenet'):
    f = open('../lib/backbone/regnet_yaml/RegNetX-32GF_dds_8gpu.yaml')
    config = yaml.load(f, Loader=yaml.FullLoader)
    model = RegNet(config)
    # print(model)
    if pretrained is not None:
        last_checkpoint = '../pretrained_models/RegNetX-32GF_dds_8gpu.pyth'
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(last_checkpoint), err_str.format(last_checkpoint)
        checkpoint = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print('have loaded checkpoint from {}'.format(last_checkpoint))
        settings = pretrained_settings['regnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


if __name__ == "__main__":
    RegNetY_200MF()
    print('')

from __future__ import print_function, division, absolute_import
from efficientnet_pytorch import EfficientNet

pretrained_settings = {
    'efficientnet': {
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
    assert num_classes == settings['num_classes'],'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def efficientnet_b0(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b0', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def efficientnet_b1(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b1', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def efficientnet_b2(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b2', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def efficientnet_b3(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b3', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def efficientnet_b4(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b4', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def efficientnet_b5(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b5', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def efficientnet_b6(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b6', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def efficientnet_b7(num_classes=1000, pretrained='imagenet'):
    model = EfficientNet.from_pretrained('efficientnet-b7', advprop=False)
    if pretrained is not None:
        settings = pretrained_settings['efficientnet'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


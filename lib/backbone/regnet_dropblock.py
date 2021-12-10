import numpy as np
import torch.nn as nn
from lib.modules.dropblock import DropBlock2D
from lib.modules.droplock_scheduler import DropLockScheduler

"""AnyNet models initalization."""


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStemIN,
        "simple_stem_in": SimpleStemIN,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None, VanillaBlock_arg=None):
        err_str = "Vanilla block does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(VanillaBlock, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=VanillaBlock_arg['BN']['EPS'], momentum=VanillaBlock_arg['BN']['MOM'])
        self.a_relu = nn.ReLU(inplace=VanillaBlock_arg['MEM']['RELU_INPLACE'])
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=VanillaBlock_arg['BN']['EPS'], momentum=VanillaBlock_arg['BN']['MOM'])
        self.b_relu = nn.ReLU(inplace=VanillaBlock_arg['MEM']['RELU_INPLACE'])

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, BasicTransform_arg, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=BasicTransform_arg['BN']['EPS'], momentum=BasicTransform_arg['BN']['MOM'])
        self.a_relu = nn.ReLU(inplace=BasicTransform_arg['MEM']['RELU_INPLACE'])
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=BasicTransform_arg['BN']['EPS'], momentum=BasicTransform_arg['BN']['MOM'])
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None, ResBasicBlock_arg=None):
        err_str = "Basic transform does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(ResBasicBlock, self).__init__()
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=ResBasicBlock_arg['BN']['EPS'], momentum=ResBasicBlock_arg['BN']['MOM'])
        self.f = BasicTransform(ResBasicBlock_arg, w_in, w_out, stride)
        self.relu = nn.ReLU(ResBasicBlock_arg['MEM']['RELU_INPLACE'])

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, SE_arg, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            nn.ReLU(inplace=SE_arg['MEM']['RELU_INPLACE']),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, BottleneckTransform_arg, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=BottleneckTransform_arg['BN']['EPS'], momentum=BottleneckTransform_arg['BN']['MOM'])
        self.a_relu = nn.ReLU(inplace=BottleneckTransform_arg['MEM']['RELU_INPLACE'])
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=BottleneckTransform_arg['BN']['EPS'], momentum=BottleneckTransform_arg['BN']['MOM'])
        self.b_relu = nn.ReLU(inplace=BottleneckTransform_arg['MEM']['RELU_INPLACE'])
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(BottleneckTransform_arg,w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=BottleneckTransform_arg['BN']['EPS'], momentum=BottleneckTransform_arg['BN']['MOM'])
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None, ResBottleneckBlock_arg=None):
        super(ResBottleneckBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=ResBottleneckBlock_arg['BN']['EPS'], momentum=ResBottleneckBlock_arg['BN']['MOM'])
        self.f = BottleneckTransform(ResBottleneckBlock_arg, w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(ResBottleneckBlock_arg['MEM']['RELU_INPLACE'])

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out, ResStemCifar_arg):
        super(ResStemCifar, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=ResStemCifar_arg['BN']['EPS'], momentum=ResStemCifar_arg['BN']['MOM'])
        self.relu = nn.ReLU(ResStemCifar_arg['MEM']['RELU_INPLACE'])

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out, ResStemIN_arg):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=ResStemIN_arg['BN']['EPS'], momentum=ResStemIN_arg['BN']['MOM'])
        self.relu = nn.ReLU(ResStemIN_arg['MEM']['RELU_INPLACE'])
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out, SimpleStemIN_arg):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=SimpleStemIN_arg['BN']['EPS'], momentum=SimpleStemIN_arg['BN']['MOM'])
        self.relu = nn.ReLU(SimpleStemIN_arg['MEM']['RELU_INPLACE'])

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r, AnyStage_arg):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r, AnyStage_arg))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet_DropBlock(nn.Module):
    """AnyNet model."""

    @staticmethod
    def get_args_(arg_any_):
        return {
            "stem_type": arg_any_['ANYNET']['STEM_TYPE'],
            "stem_w": arg_any_['ANYNET']['STEM_W'],
            "block_type": arg_any_['ANYNET']['BLOCK_TYPE'],
            "ds": arg_any_['ANYNET']['DEPTHS'] ,
            "ws": arg_any_['ANYNET']['WIDTHS'],
            "ss": arg_any_['ANYNET']['STRIDES'] ,
            "bms": arg_any_['ANYNET']['BOT_MULS']  ,
            "gws": arg_any_['ANYNET']['GROUP_WS'],
            "se_r": arg_any_['ANYNET']['SE_R']if  arg_any_['ANYNET']['SE_ON'] else None,
            "nc": arg_any_['MODEL']['NUM_CLASSES'],
        }

    def __init__(self, kwargs_any, arg_any):
        super(AnyNet_DropBlock, self).__init__()
        kwargs_ = self.get_args_(arg_any) if not kwargs_any else kwargs_any
        self._construct(kwargs_, arg_any)

    def _construct(self, kwargs_, arg_any):
        # Generate dummy bot muls and gs for models that do not use them
        stem_type = kwargs_['stem_type']
        stem_w = kwargs_['stem_w']
        block_type = kwargs_['block_type']
        ds = kwargs_['ds']
        ws = kwargs_['ws']
        ss = kwargs_['ss']
        bms = kwargs_['bms']
        gws = kwargs_['gws']
        se_r = kwargs_['se_r']
        nc = kwargs_['nc']
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w, arg_any)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w

        s_len = len(stage_params)
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r, arg_any))
            # add drop block
            name = "s{}_drop_block".format(i + 1)
            if i >= s_len - 2:      # only last block, use drop_block
                self.add_module(name, DropLockScheduler(DropBlock2D()))
            prev_w = w

        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def dropblock_step(self):
        for module in self.modules():
            if isinstance(module, DropLockScheduler):
                module.step()       # increment number of iterations

    def init_dropblock_para(self, start=0.0, stop=0.1, block_size=5, nr_steps=5000):
        for module in self.modules():
            if isinstance(module, DropLockScheduler):
                module.init_para(start, stop, block_size, nr_steps)

    def init_dropout_para(self, drop_prob=0.5):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_prob


"""RegNet models initalization."""


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet_DropBlock(AnyNet_DropBlock):
    """RegNet model."""

    @staticmethod
    def get_args(args_):
        """Convert RegNet to AnyNet parameter format."""
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = args_['REGNET']['WA'], args_['REGNET']['W0'], args_['REGNET']['WM'], args_['REGNET']['DEPTH']
        ws, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        # Convert to per stage format
        s_ws, s_ds = get_stages_from_blocks(ws, ws)
        # Use the same gw, bm and ss for each stage
        s_gs = [args_['REGNET']['GROUP_W']  for _ in range(num_stages)]
        s_bs = [args_['REGNET']['BOT_MUL']  for _ in range(num_stages)]
        s_ss = [args_['REGNET']['STRIDE']  for _ in range(num_stages)]
        # Adjust the compatibility of ws and gws
        s_ws, s_gs = adjust_ws_gs_comp(s_ws, s_bs, s_gs)
        # Get AnyNet arguments defining the RegNet
        return {
            "stem_type": args_['REGNET']['STEM_TYPE'],
            "stem_w": args_['REGNET']['STEM_W'],
            "block_type": args_['REGNET']['BLOCK_TYPE'],
            "ds": s_ds,
            "ws": s_ws,
            "ss": s_ss,
            "bms": s_bs,
            "gws": s_gs,
            "se_r": args_['REGNET']['SE_R'] if args_['REGNET']['SE_ON'] else None,
            "nc": args_['MODEL']['NUM_CLASSES'],
            # "args":args_,
        }

    def __init__(self, args):
        kwargs = RegNet_DropBlock.get_args(args)
        super(RegNet_DropBlock, self).__init__(kwargs, args)


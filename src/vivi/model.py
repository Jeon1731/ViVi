import torch.nn as nn
import math


__all__ = ['mobilefacenetv3_large', 'mobilefacenetv3_small']

def _make_divisible(v, divisor, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True): # 입력받은 텐서의 데이터를 새로 복사하지 않고 그 자리(In-place)에서 직접 수정
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # squeeze
        self.fc = nn.Sequential( # excitation
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            # nn.ReLU(inplace=True),
            nn.PReLU(), # (2) ReLU -> PReLU
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size, padding, groups=inp, bias=bias)
        self.pointwise = nn.Conv2d(inp, oup, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(inp)
        self.bn2 = nn.BatchNorm2d(oup)
        # self.relu = nn.ReLU()
        self.prelu = nn.PReLU() # (2) ReLU -> PReLU

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.prelu(x) # (2) ReLU -> PReLU

        x = self.pointwise(x)
        x = self.bn2(x)
        # x = self.relu(x)
        x = self.prelu(x) # (2) ReLU -> PReLU

        return x

class GDConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, oup, kernel_size, padding=padding, groups=inp, bias=bias)
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in  [1, 2]

        self.identity = stride == 1 and inp == oup # 잔차연결 가능

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size -1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # squeeze-and-excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pointwise-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # squeeze-and-excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pointwise-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity: # 잔차연결 가능
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileFaceNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=2, embedding_dim=128, width_mult=1):
        super(MobileFaceNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = DepthwiseSeparableConv
        layers.append(block(inp=24, oup=24, kernel_size=3, padding=1)) # (3) fase downsampling: bottleneck -> depthwise-conv3x3
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.gdconv = GDConv(inp=exp_size, oup=exp_size, kernel_size=14, padding=0)
        output_channel = {'large' : 1280, 'small' : 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.embedding = nn.Linear(exp_size, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.gdconv(x).flatten(1)
        classification_logits = self.classifier(x)
        embedding = self.embedding(x)
        return classification_logits, embedding

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilefacenetv3_large(**kwargs):
    """
    Constructs a MobileFaceNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    cfgs_vivi = [
        # k, t, c, SE, HS, s 
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   2, 112, 1, 1, 1], # (4) smaller expansion factor
        [3,   4, 112, 1, 1, 1], # (4) smaller expansion factor
        [5,   2, 160, 1, 1, 2], # (4) smaller expansion factor
        [5,   4, 160, 1, 1, 1], # (4) smaller expansion factor
        [5,   2, 160, 1, 1, 1]  # (4) smaller expansion factor
    ]
    return MobileFaceNetV3(cfgs_vivi, mode='large', **kwargs)

def mobilefacenetv3_small(**kwargs):
    """
    Constructs a MobileFaceNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    return MobileFaceNetV3(cfgs, mode='small', **kwargs) 
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import para
from model_utils import Decoder, Header, conv3x3

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_dilation=False, downsample=None, use_bn=True):
        super(Bottleneck, self).__init__()
        bias = not use_bn
        self.use_bn = use_bn
        dilation = 1
        padding = 1
        if use_dilation:
            assert stride == 1, 'stride != 1 when dilation is True'
            dilation = 2
            padding = 2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                dilation=dilation, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = F.relu(residual + out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BackBone(nn.Module):

    def __init__(self, block, num_block, input_channels, use_bn=True):
        super(BackBone, self).__init__()

        self.use_bn = use_bn

        # Block 1
        self.conv1 = conv3x3(input_channels, 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        if para.use_se_mod:
            self.se1 = SELayer(32)

        # Block 2
        self.in_planes = 32
        self.block2 = self._make_layer(block, 24, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.conv1(x)
        if para.use_se_mod:
            x = self.se1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p5 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p4 = l3 + self.deconv2(p5)

        return p4

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        # x2
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample))
        # x1
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, use_dilation=False))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y


class PIXOR(nn.Module):
    '''
    The input of PIXOR nn module is a tensor of [batch_size, height, weight, channel]
    The output of PIXOR nn module is also a tensor of [batch_size, height/4, weight/4, channel]
    Note that we convert the dimensions to [C, H, W] for PyTorch's nn.Conv2d functions
    '''

    def __init__(self, use_bn=True, input_channels=36, decode=False):
        super(PIXOR, self).__init__()
        self.backbone = BackBone(Bottleneck, [3, 6, 6, 3], input_channels, use_bn)
        self.header = Header(use_bn)
        self.corner_decoder = Decoder()
        self.use_decode = decode


    def set_decode(self, decode):
        self.use_decode = decode

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # Torch Takes Tensor of shape (Batch_size, channels, height, width)

        features = self.backbone(x)
        cls, reg = self.header(features)

        if self.use_decode:
            if para.estimate_bh:
                reg, bh = self.corner_decoder(reg)
            else:
                reg = self.corner_decoder(reg)

        # Return tensor(Batch_size, height, width, channels)
        cls = cls.permute(0, 2, 3, 1)
        reg = reg.permute(0, 2, 3, 1)

        if self.use_decode and para.estimate_bh:
            return torch.cat([cls, reg], dim=3), bh.permute(0, 2, 3, 1)
        else:
            return torch.cat([cls, reg], dim=3)

###################################################################################################

class BackBone_RFB(nn.Module):
    def __init__(self, bottleneck, rfb, num_block, input_channels, use_bn=True):
        super(BackBone_RFB, self).__init__()
        self.use_bn = use_bn
        self.bottleneck = bottleneck
        self.rfbblock = rfb

        # Block 1
        self.conv1 = conv3x3(input_channels, 32)
        self.conv2 = conv3x3(32, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        if para.use_se_mod:
            self.se1 = SELayer(32)

        # Block 2
        self.in_planes = 32
        self.block2 = self._make_layer(24, num_blocks=num_block[0])
        self.block3 = self._make_layer(48, num_blocks=num_block[1])
        self.block4 = self._make_layer(64, num_blocks=num_block[2])
        self.block5 = self._make_layer(96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.conv1(x)
        if para.use_se_mod:
            x = self.se1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        p5 = l4 + self.deconv1(l5)
        l3 = self.latlayer3(c3)
        p4 = l3 + self.deconv2(p5)

        return p4

    def _make_layer(self, planes, num_blocks):
        expansion = self.bottleneck.expansion
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        # x2
        layers.append(self.bottleneck(self.in_planes, planes, stride=2, downsample=downsample))
        # x1
        self.in_planes = planes * expansion
        layers.append(self.rfbblock(self.in_planes, self.in_planes))
        for _ in range(num_blocks):
            layers.append(self.bottleneck(self.in_planes, planes, use_dilation=False))
        return nn.Sequential(*layers)

class PIXOR_RFB(nn.Module):
    '''
    The input of PIXOR nn module is a tensor of [batch_size, height, weight, channel]
    The output of PIXOR nn module is also a tensor of [batch_size, height/4, weight/4, channel]
    Note that we convert the dimensions to [C, H, W] for PyTorch's nn.Conv2d functions
    '''
    def __init__(self, use_bn=True, input_channels=36, decode=False):
        super(PIXOR_RFB, self).__init__()
        self.backbone = BackBone_RFB(Bottleneck, BasicRFB, [2, 2, 2, 2], input_channels, use_bn)
        self.header = Header(use_bn)
        self.corner_decoder = Decoder()
        self.use_decode = decode

    def set_decode(self, decode):
        self.use_decode = decode

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        features = self.backbone(x)
        cls, reg = self.header(features)

        if self.use_decode:
            if para.estimate_bh:
                reg, bh = self.corner_decoder(reg)
            else:
                reg = self.corner_decoder(reg)

        # Return tensor(Batch_size, height, width, channels)
        cls = cls.permute(0, 2, 3, 1)
        reg = reg.permute(0, 2, 3, 1)

        if self.use_decode and para.estimate_bh:
            return torch.cat([cls, reg], dim=3), bh.permute(0, 2, 3, 1)
        else:
            return torch.cat([cls, reg], dim=3)

class BasicConvRFB(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConvRFB, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConvRFB(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConvRFB(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConvRFB(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConvRFB(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConvRFB(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConvRFB(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConvRFB(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConvRFB((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConvRFB(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConvRFB(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConvRFB(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

###################################################################################################

def test():
    print("Testing PIXOR model")
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    net = PIXOR(use_bn=False).to(dev)
    preds = net(torch.autograd.Variable(torch.randn(2, 800, 700, 36).to(dev)))
    print("Prediction output size", preds.size()) # [2, 200, 175, 7]

def test_decoder():
    print("Testing PIXOR decoder")
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    net = PIXOR(use_bn=False).to(dev)
    net.set_decode(True)
    preds = net(torch.autograd.Variable(torch.randn(2, 800, 700, 36).to(dev)))
    print("Predictions output size", preds.size()) # [2, 200, 175, 9]

def test_bottleneck():
    print("Testing PIXOR bottleneck")
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    in_planes = 3 * Bottleneck.expansion
    planes = 3
    net = Bottleneck(in_planes, planes, use_dilation=True).to(dev)
    preds = net(torch.autograd.Variable(torch.randn(2, in_planes, 800, 700).to(dev)))
    print("Predictions output size1", preds.size()) # [2, 12, 800, 700]
    #
    in_planes = 5
    planes = 7
    downsample = nn.Conv2d(in_planes, planes * Bottleneck.expansion,
                           kernel_size=1, stride=2, bias=True)
    net = Bottleneck(in_planes, planes, stride=2, downsample=downsample).to(dev)
    preds = net(torch.autograd.Variable(torch.randn(2, in_planes, 800, 700).to(dev)))
    print("Predictions output size2", preds.size()) # [2, 28, 400, 350]

def test_backbone():
    print("Testing PIXOR backbone")
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    net = BackBone(Bottleneck, [3, 6, 6, 3], 36, use_bn=True).to(dev)
    preds = net(torch.autograd.Variable(torch.randn(2, 36, 800, 700).to(dev)))
    print("Predictions output size", preds.size()) # [2, 96, 200, 175]

def test_header():
    print("Testing PIXOR header")
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    net = Header(use_bn=True).to(dev)
    cls, pred = net(torch.autograd.Variable(torch.randn(2, 96, 200, 175).to(dev)))
    print("Predictions output size", cls.size(), pred.size()) # [2, 1, 200, 175], [2, 6, 200, 175]

def test_basic_rfb():
    print("Testing PIXOR header")
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    net = BasicRFB(96, 128).to(dev)
    preds = net(torch.autograd.Variable(torch.randn(2, 96, 200, 175).to(dev)))
    print("Predictions output size", preds.size()) # [2, 128, 200, 175]

def test_pixor_rfb():
    print("Testing PIXOR_RFB model")
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    net = PIXOR_RFB(use_bn=False, input_channels=36).to(dev)
    net.set_decode(True)
    preds = net(torch.autograd.Variable(torch.randn(2, 800, 700, 36).to(dev)))
    print("Predictions output size", preds.size()) # [2, 200, 175, 9]

if __name__ == "__main__":
    fire.Fire()

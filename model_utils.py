import torch
from torch import nn
from params import para

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.geometry = [para.L1, para.L2, para.W1, para.W2]
        self.grid_size = para.grid_sizeLW * para.ratio

        self.target_mean = para.target_mean
        self.target_std_dev = para.target_std_dev

    def forward(self, x):
        '''

        :param x: Tensor 6-channel geometry
        6 channel map of [cos(yaw), sin(yaw), log(x), log(y), w, l]
        Shape of x: (B, C=6, H=200, W=175)
        :return: Concatenated Tensor of 8 channel geometry map of bounding box corners
        8 channel are [rear_left_x, rear_left_y,
                        rear_right_x, rear_right_y,
                        front_right_x, front_right_y,
                        front_left_x, front_left_y]
        Return tensor has a shape (B, C=8, H=200, W=175), and is located on the same device as x

        '''
        # Tensor in (B, C, H, W)

        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()

        for i in range(para.box_code_len):
            x[:, i, :, :] = x[:, i, :, :] * self.target_std_dev[i] + self.target_mean[i]

        if para.box_code_len == 6:
            cos_t, sin_t, dx, dy, log_w, log_l = torch.chunk(x, 6, dim=1)
            theta = torch.atan2(sin_t, cos_t)
        elif para.box_code_len == 5:
            theta, dx, dy, log_w, log_l = torch.chunk(x, 5, dim=1)
        elif para.box_code_len == 8:
            cos_t, sin_t, dx, dy, log_w, log_l, log_bottom, log_head = torch.chunk(x, 8, dim=1)
            theta = torch.atan2(sin_t, cos_t)
        elif para.box_code_len == 7:
            theta, dx, dy, log_w, log_l, log_bottom, log_head = torch.chunk(x, 7, dim=1)

        if para.sin_angle_loss:
            theta = torch.asin(theta)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x = torch.arange(self.geometry[2], self.geometry[3], self.grid_size, dtype=torch.float32, device=device)
        y = torch.arange(self.geometry[0], self.geometry[1], self.grid_size, dtype=torch.float32, device=device)
        x = x[:para.label_shape[1]]
        y = y[:para.label_shape[0]]

        yy, xx = torch.meshgrid([y, x])
        centre_y = yy + dy
        centre_x = xx + dx
        l = log_l.exp()
        w = log_w.exp()
        rear_left_x = centre_x - l/2 * cos_t - w/2 * sin_t
        rear_left_y = centre_y - l/2 * sin_t + w/2 * cos_t
        rear_right_x = centre_x - l/2 * cos_t + w/2 * sin_t
        rear_right_y = centre_y - l/2 * sin_t - w/2 * cos_t
        front_right_x = centre_x + l/2 * cos_t + w/2 * sin_t
        front_right_y = centre_y + l/2 * sin_t - w/2 * cos_t
        front_left_x = centre_x + l/2 * cos_t - w/2 * sin_t
        front_left_y = centre_y + l/2 * sin_t + w/2 * cos_t

        decoded_reg = torch.cat([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                                 front_right_x, front_right_y, front_left_x, front_left_y], dim=1)

        if para.estimate_bh:
            log_bottom = log_bottom.exp() - para.height_bias
            log_head = log_head.exp() - para.height_bias
            return decoded_reg, torch.cat([log_bottom, log_head], dim=1)
        else:
            return decoded_reg


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class Header(nn.Module):

    def __init__(self, use_bn=True, input_channels=96, aggr_feat=False):
        super(Header, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn
        self.aggr_feat = aggr_feat
        if aggr_feat:
            self.conv1 = conv3x3(input_channels, 96, bias=bias)
            self.bn1 = nn.BatchNorm2d(96)
            self.conv2 = conv3x3(96, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
            self.conv3 = conv3x3(96, 96, bias=bias)
            self.bn3 = nn.BatchNorm2d(96)
            self.conv4 = conv3x3(96, 96, bias=bias)
            self.bn4 = nn.BatchNorm2d(96)
            channels = 96
        else:
            channels = input_channels

        self.clshead = conv3x3(channels, 1, bias=True)
        if para.sin_angle_loss:
            self.anglehead = conv3x3(channels, 1, bias=True)
            self.reghead = conv3x3(channels, para.box_code_len-1, bias=True)
        else:
            self.reghead = conv3x3(channels, para.box_code_len, bias=True)

    def forward(self, x):
        if self.aggr_feat:
            x = self.conv1(x)
            if self.use_bn:
                x = self.bn1(x)
            x = self.conv2(x)
            if self.use_bn:
                x = self.bn2(x)
            x = self.conv3(x)
            if self.use_bn:
                x = self.bn3(x)
            x = self.conv4(x)
            if self.use_bn:
                x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        if para.sin_angle_loss:
            angle = torch.tanh(self.anglehead(x))
            reg = torch.cat([angle, reg], dim=1)

        return cls, reg

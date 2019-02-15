import numpy as np
import fire
from collections import OrderedDict
import inspect
import torch
from torch import nn
from torch.nn import functional as F
import spconv

from params import para, use_dense_net
from model_utils import Decoder, Header, conv3x3

class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw

def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # i = 0
        for module in self._modules.values():
            # print(i)
            input = module(input)
            # i += 1
        return input

#######################################################################################

class SpMiddleFHD(nn.Module):
    def __init__(self,
                 dense_shape,
                 ratio,
                 use_norm=True,
                 name='SpMiddleFHD'):
        super(SpMiddleFHD, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
        self.ratio = ratio
        if self.ratio == 8:
            self.sparse_shape = np.array(dense_shape[1:4]) + [1, 0, 0]
            # input: # [1600, 1400, 41]
            self.middle_conv = spconv.SparseSequential(
                SubMConv3d(para.voxel_feature_len, 16, 3, indice_key="subm0", dilation=1),
                BatchNorm1d(16),
                nn.ReLU(),
                SubMConv3d(16, 16, 3, indice_key="subm0", dilation=2),
                BatchNorm1d(16),
                nn.ReLU(),
                SpConv3d(16, 32, 3, 2, padding=1), # [1600, 1400, 41] -> [800, 700, 21]
                BatchNorm1d(32),
                nn.ReLU(),
                #
                SubMConv3d(32, 32, 3, indice_key="subm1", dilation=1),
                BatchNorm1d(32),
                nn.ReLU(),
                SubMConv3d(32, 32, 3, indice_key="subm1", dilation=2),
                BatchNorm1d(32),
                nn.ReLU(),
                SpConv3d(32, 64, 3, 2, padding=1), # [800, 700, 21] -> [400, 350, 11]
                BatchNorm1d(64),
                nn.ReLU(),
                #
                SubMConv3d(64, 64, 3, indice_key="subm2", dilation=1),
                BatchNorm1d(64),
                nn.ReLU(),
                SubMConv3d(64, 64, 3, indice_key="subm2", dilation=2),
                BatchNorm1d(64),
                nn.ReLU(),
                SubMConv3d(64, 64, 3, indice_key="subm2", dilation=3),
                BatchNorm1d(64),
                nn.ReLU(),
                #
                SpConv3d(64, 64, 3, 2, padding=[0, 1, 1]),  # [400, 350, 11] -> [200, 175, 5]
                BatchNorm1d(64),
                nn.ReLU(),
                SubMConv3d(64, 64, 3, indice_key="subm3", dilation=1),
                BatchNorm1d(64),
                nn.ReLU(),
                SubMConv3d(64, 64, 3, indice_key="subm3", dilation=2),
                BatchNorm1d(64),
                nn.ReLU(),
                SubMConv3d(64, 64, 3, indice_key="subm3", dilation=3),
                BatchNorm1d(64),
                nn.ReLU(),
                SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),  # [200, 175, 5] -> [200, 175, 2]
                BatchNorm1d(64),
                nn.ReLU(),
            )
        elif self.ratio == 4:
            self.sparse_shape = np.array(dense_shape[1:4])
            # input: # [800, 700, 40]
            self.middle_conv = spconv.SparseSequential(
                SubMConv3d(para.voxel_feature_len, 32, 3, indice_key="subm0", dilation=1),
                BatchNorm1d(32),
                nn.ReLU(),
                SubMConv3d(32, 32, 3, indice_key="subm0", dilation=2),
                BatchNorm1d(32),
                nn.ReLU(),
                SubMConv3d(32, 32, 3, indice_key="subm0", dilation=3),
                BatchNorm1d(32),
                nn.ReLU(),
                SpConv3d(32, 32, 3, 2, padding=1), # [800, 700, 40] -> [400, 350, 20]
                BatchNorm1d(32),
                nn.ReLU(),
                #
                SubMConv3d(32, 64, 3, indice_key="subm1", dilation=1),
                BatchNorm1d(64),
                nn.ReLU(),
                SubMConv3d(64, 64, 3, indice_key="subm1", dilation=2),
                BatchNorm1d(64),
                nn.ReLU(),
                SubMConv3d(64, 64, 3, indice_key="subm1", dilation=3),
                BatchNorm1d(64),
                nn.ReLU(),
                SpConv3d(64, 64, 3, 2, padding=1), # [400, 350, 20] -> [200, 175, 10]
                BatchNorm1d(64),
                nn.ReLU(),
                #
                SpConv3d(128, 128, (3, 1, 1)),  # [200, 175, 10] -> [200, 175, 8]
                BatchNorm1d(128),
                nn.ReLU(),

                SpConv3d(128, 128, (3, 1, 1)),  # [200, 175, 4] -> [200, 175, 2]
                BatchNorm1d(128),
                nn.ReLU(),

                SpConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1)),  # [200, 175, 8] -> [200, 175, 4]
                BatchNorm1d(128),
                nn.ReLU(),

                SpConv3d(128, 128, (2, 1, 1)),  # [200, 175, 2] -> [200, 175, 1]
                BatchNorm1d(128),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        print('sparse_shape', self.sparse_shape)

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # t = time.time()
        # torch.cuda.synchronize()
        ret = self.middle_conv(ret)
        # torch.cuda.synchronize()
        # print("spconv forward time", time.time() - t)
        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class RPNV2(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 decode=False,
                 num_class=1,
                 layer_nums=[5, 5],
                 layer_strides=[1, 2],
                 num_filters=[128, 256],
                 upsample_strides=[1, 2],
                 num_upsample_filters=[256, 256],
                 num_input_features=128,
                 name='rpn'):
        super(RPNV2, self).__init__()

        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)

        if use_norm:
            BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(
                    in_filters[i], num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    num_filters[i],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        self.header = Header(input_channels=sum(num_upsample_filters), use_bn=use_norm)

    def forward(self, x, bev=None):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        cls, reg = self.header(x)

        return cls, reg

def voxel_feature_extractor(features, num_voxels):
    # features: [concated_num_points, num_voxel_size, 3(4)]
    # num_voxels: [concated_num_points]
    if para.voxel_feature_len == 2:
        points_mean = features[:, :, 2:4].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
    elif para.voxel_feature_len == 4:
        points_mean = features[:, :, :4].sum(dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
    return points_mean.contiguous()

class PIXOR_SPARSE(nn.Module):
    def __init__(self, full_shape, ratio, use_bn=True, decode=False, num_input_features=128):
        super(PIXOR_SPARSE, self).__init__()
        self.use_decode = decode
        self.full_shape = full_shape
        print('full_shape', full_shape)
        dense_shape = [1] + full_shape[::-1].tolist() + [num_input_features]
        self.middle_feature_extractor = SpMiddleFHD(dense_shape, ratio, use_norm=use_bn)
        self.rpn = RPNV2(use_norm=use_bn)
        self.corner_decoder = Decoder()

    def set_decode(self, decode):
        self.use_decode = decode

    def forward(self, voxels_feature, coords_pad, batch_size):
        dense_map = self.middle_feature_extractor(voxels_feature, coords_pad, batch_size)
        cls, reg = self.rpn(dense_map)

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

def _prepare_voxel(dev):
    # generate voxel features
    from voxel_gen import VoxelGenerator
    voxel_generator = VoxelGenerator(
        voxel_size=[para.grid_sizeLW, para.grid_sizeLW, para.grid_sizeH],
        point_cloud_range=[para.W1, para.L1, para.H1,
                           para.W2, para.L2, para.H2],
        max_num_points=30,
        max_voxels=40000
    )
    points = np.random.rand(40000, 4).astype(np.float32) * 50
    # X,Y,Z  ->  Z,Y,X
    voxels, coords, num_points = voxel_generator.generate(points)
    coords_pad = np.pad(
        coords, ((0, 0), (1, 0)),
        mode='constant',
        constant_values=0)

    voxels = torch.tensor(voxels, dtype=torch.float32, device=dev)       # (M, K, 4)
    coords_pad = torch.tensor(coords_pad, dtype=torch.int32, device=dev) # (M, 3+1)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=dev) # (M,)
    voxels_feature = voxel_feature_extractor(voxels, num_points) # (M, C)
    grid_size = voxel_generator.grid_size

    return voxels_feature, coords_pad, grid_size

def test():
    use_dense_net(False)
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    voxels_feature, coords_pad, grid_size = _prepare_voxel(dev)

    batch_size_dev = 1
    # generate middle features
    dense_shape = [1] + grid_size[::-1].tolist() + [128]
    print('dense_shape', dense_shape)
    middle_feature_extractor = SpMiddleFHD(dense_shape, para.ratio).to(dev)
    dense_map = middle_feature_extractor(voxels_feature, coords_pad, batch_size_dev)
    print('dense_map', dense_map.size())

    rpn = RPNV2().to(dev)
    cls, reg = rpn(dense_map)
    print('rpn_ret', cls.size(), reg.size())

def test1():
    use_dense_net(False)
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    voxels_feature, coords_pad, grid_size = _prepare_voxel(dev)

    net = PIXOR_SPARSE(grid_size, para.ratio, use_bn=True, decode=True).to(dev)
    out = net(voxels_feature, coords_pad, batch_size=1)
    if net.use_decode and para.estimate_bh:
        print('out', out[0].size(), out[1].size())
    else:
        print('out', out.size())

if __name__ == '__main__':
    fire.Fire()

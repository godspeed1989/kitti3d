import torch
import torch.nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os.path
import numba
import ipdb
import torch.optim as optim
import torch.nn as nn
from functools import partial

from train_utils.fastai_optim import OptimWrapper
from train_utils import learning_schedules_fastai as lsf

from params import para
from model import PIXOR, PIXOR_RFB
from model_sp import PIXOR_SPARSE
from loss import CustomLoss, MultiTaskLoss, GHM_Loss

def dict2str(dt):
    res = ''
    for key, val in dt.items():
        if callable(val):
            v = val.__name__
        else:
            v = str(val)
        res += '%20s: %s\n' % (str(key), str(v))
    return res

def trasform_label2metric(label, ratio=4, grid_size=0.1, base_height=400):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in label map space
    :return: numpy array of shape [..., 2] of the same coordinates in metric space
    '''
    metric = np.copy(label)
    metric[..., 1] -= base_height
    metric = metric * grid_size * ratio
    return metric

def transform_metric2label(metric, ratio=4, grid_size=0.1, base_height=400):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in metric space
    :return: numpy array of shape [..., 2] of the same coordinates in label_map space
    '''
    label = (metric / ratio) / grid_size
    label[..., 1] += base_height
    return label

def plot_bev(velo_array, predict_list=None, label_list=None, window_name='GT'):
    '''
    Plot a Birds Eye View Lidar and Bounding boxes (Using OpenCV!)
    The heading of the vehicle is marked as a red line
        (which connects front right and front left corner)

    :param velo_array: a 2d velodyne points
    :param label_list: a list of numpy arrays of shape [4, 2], which corresponds to the 4 corners' (x, y)
    The corners should be in the following sequence:
    rear left, rear right, front right and front left
    :param window_name: name of the open_cv2 window
    :return: None
    '''
    intensity = np.zeros((velo_array.shape[0], velo_array.shape[1], 3), dtype=np.uint8)
    val = velo_array[:, :, :].max(axis=2, keepdims=True)
    val = 1 - val / (np.ptp(val) + 1e-5)
    intensity[:, :, :] = (val * 255).astype(np.uint8)

    intensity1 = intensity.copy()
    intensity2 = intensity.copy()
    if label_list is not None:
        for corners in label_list:
            plot_corners = corners / para.grid_sizeLW
            plot_corners[:, 1] += int(para.input_shape[0]//2)
            plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(intensity1, [plot_corners], True, (0, 0, 255), 2)
            cv2.line(intensity1, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 255, 0), 2)
    if predict_list is not None:
        for corners in predict_list:
            plot_corners = corners / para.grid_sizeLW
            plot_corners[:, 1] += int(para.input_shape[0]//2)
            plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(intensity2, [plot_corners], True, (255, 255, 0), 2)
            cv2.line(intensity2, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (255, 0, 0), 2)

    # ipdb.set_trace()
    intensity1 = intensity1.astype(np.uint8)
    intensity2 = intensity2.astype(np.uint8)
    if intensity.shape[0] > 1000:
        scale = intensity.shape[0] / 800.0
        height, width = intensity.shape[:2]
        dsize = (int(width / scale), int(height / scale))
        intensity1 = cv2.resize(intensity1, dsize, interpolation=cv2.INTER_AREA)
        intensity2 = cv2.resize(intensity2, dsize, interpolation=cv2.INTER_AREA)

    alpha = 0.5
    cv2.addWeighted(intensity1, alpha, intensity2, 1 - alpha, 0, intensity)
    cv2.imshow(window_name, intensity)
    cv2.imwrite(window_name+'.png', intensity)

def plot_label_map(label_map):
    plt.figure()
    plt.imshow(label_map)
    plt.show()

def get_points_in_a_rotated_box(corners, xmin = 0, ymin = 0, xmax = 176, ymax = 200):
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y0 is lowest
            return int(math.floor(y0))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # lowest point is at left edge of pixel column
            return int(math.floor(y0 + m * (x - x0)))
        else:
            # lowest point is at right edge of pixel column
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))


    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y1 is highest
            return int(math.ceil(y1))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # highest point is at right edge of pixel column
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            # highest point is at left edge of pixel column
            return int(math.ceil(y0 + m * (x - x0)))


    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)

        elif xf < m2x:
            if m1y < m2y:
                # Phase IIa: left/bottom --> top/right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            else:
                # Phase IIb: left/top --> bottom/right
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

        else:
            # Phase III: bottom/top --> right
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)

        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels

def load_config(path):
    """ Loads the configuration file

     Args:
         path: A string indicating the path to the configuration file
     Returns:
         config: A Python dictionary of hyperparameter name-value pairs
         learning rate: The learning rate of the optimzer
         batch_size: Batch size used during training
         max_epochs: Number of epochs to train the network for
     """
    with open(path) as file:
        config = json.load(file)

    learning_rate = config["learning_rate"]
    batch_size = para.batch_size
    max_epochs = config["max_epochs"]

    return config, learning_rate, batch_size, max_epochs

def get_model_name(name, config, para):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        name: Name of ckpt
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    code_len = para.box_code_len
    folder = "experiments_{}_c{}".format(para.channel_type, code_len)
    if para.use_se_mod:
        folder += "_se"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if name is not None:
        path = os.path.join(folder, name)
    else:
        file_list = os.listdir(folder)
        prefix_len = len(para.net) + len("__epoch")
        file_list.sort(key = lambda x: int(x[prefix_len:]))
        path = os.path.join(folder, file_list[-1])
    return path

@numba.jit(nopython=False)
def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces

@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d

def remove_points_in_boxes(points, rbbox_corners):
    '''
    points (N, 3/4)
    rbbox_corners (N, 8, 3)
    '''
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    points = points[np.logical_not(indices.any(-1))]
    return points, np.logical_not(indices.any(-1))

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):
    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

def bnm_lmbd(cur_epoch):
    cur_decay = 1
    BN_DECAY_STEP_LIST = [650]
    BN_DECAY = 0.5
    BN_MOMENTUM = 0.1
    BNM_CLIP = 0.01
    for decay_step in BN_DECAY_STEP_LIST:
        if cur_epoch >= decay_step:
            cur_decay = cur_decay * BN_DECAY
    return max(BN_MOMENTUM * cur_decay, BNM_CLIP)

def build_model(config, device, train=True, train_loader=None):
    if para.net == 'PIXOR':
        net = PIXOR(use_bn=config['use_bn'], input_channels=para.input_channels).to(device)
    elif para.net == 'PIXOR_RFB':
        net = PIXOR_RFB(use_bn=config['use_bn'], input_channels=para.input_channels).to(device)
    elif para.net == 'PIXOR_SPARSE':
        net = PIXOR_SPARSE(para.full_shape, para.ratio, use_bn=config['use_bn']).to(device)
    else:
        raise NotImplementedError

    if config['loss_type'] == "MultiTaskLoss":
        criterion = MultiTaskLoss(device=device, num_classes=1)
    elif config['loss_type'] == "CustomLoss":
        criterion = CustomLoss(device=device, num_classes=1)
    elif config['loss_type'] == "GHM_Loss":
        criterion = GHM_Loss(device=device, num_classes=1)
    else:
        raise NotImplementedError

    if not train:
        return net, criterion

    if config['optimizer'] == 'ADAM':
        optimizer = torch.optim.Adam(params=[{'params': criterion.parameters()},
                                             {'params': net.parameters()}],
                                     lr=config['learning_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    elif config['optimizer'] == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(net), wd=0.001, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    if config['optimizer'] == 'adam_onecycle':
        total_steps = len(train_loader) * config['max_epochs']
        scheduler = lsf.OneCycle(
            optimizer, total_steps, config['learning_rate'], list([0.95, 0.85]), 10.0, 0.4)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_decay_every'], gamma=0.5)

    bnm_scheduler = BNMomentumScheduler(net, bnm_lmbd, last_epoch=-1)

    return net, criterion, optimizer, scheduler, bnm_scheduler

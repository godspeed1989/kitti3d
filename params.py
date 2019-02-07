import numpy as np
from easydict import EasyDict

para = EasyDict()

# 6: cos, sin, x, y, w, l
# 5: r, x, y, w, l
para.box_code_len = 6
if para.box_code_len == 6:
    para.target_mean = np.array([0.022, -0.006,  0.194,  0.192,  0.487,  1.37], dtype=np.float32)
    para.target_std_dev = np.array([1.0, 1.0, 0.537, 0.389, 0.064, 0.109], dtype=np.float32)
elif para.box_code_len == 5:
    para.target_mean = np.array([0.262, 0.194, 0.192, 0.487, 1.37 ], dtype=np.float32)
    para.target_std_dev = np.array([1.0, 0.537, 0.389, 0.064, 0.109], dtype=np.float32)
else:
    raise NotImplementedError

para.L1 = -40.0
para.L2 = 40.0
para.W1 = 0.0
para.W2 = 70.4
para.H1 = -3
para.H2 = 1.0

para.dense_net = False
def use_dense_net(sel):
    para.dense_net = sel
    if para.dense_net:
        para.grid_sizeLW = 0.1
        para.grid_sizeH = 0.1
        para.ratio = 4
        para.input_shape = (800, 704)
        # PIXOR or PIXOR_RFB
        para.net = 'PIXOR'
        # 'rgb', 'pixor', 'pixor-rgb', 'voxel'
        para.channel_type = 'rgb'
        para.batch_size = 4
    else:
        para.grid_sizeLW = 0.1
        para.grid_sizeH = 0.1
        if para.grid_sizeLW == 0.05:
            para.ratio = 8
            para.input_shape = (1600, 1408)
            para.full_shape = np.array([1408, 1600, 40])
            para.batch_size = 6
        elif para.grid_sizeLW == 0.1:
            para.ratio = 4
            para.input_shape = (800, 704)
            para.full_shape = np.array([704, 800, 40])
            para.batch_size = 6
        para.net = 'PIXOR_SPARSE'
        para.channel_type = 'sparse'
use_dense_net(para.dense_net)

para.label_shape = (200, 176)

if para.channel_type == 'rgb':
    para.input_channels = 3
if para.channel_type == 'pixor':
    para.input_channels = int((para.H2 - para.H1) / para.grid_sizeH + 1)
if para.channel_type == 'pixor-rgb':
    para.input_channels = int(3 + (para.H2 - para.H1) / para.grid_sizeH + 1)
if para.channel_type == 'voxel':
    para.input_channels = int((para.H2 - para.H1) / para.grid_sizeH)
if para.channel_type == 'sparse':
    para.input_channels = -1

para.object_list = ['Car']

para.box_in_labelmap_ratio = 0.6

para.use_se_mod = False

para.align_pc_with_img = False
para.img_shape = (375, 1242)
para.crop_pc_by_fov = True

para.augment_data_use_db = True
para.augment_max_samples = 8
para.remove_points_after_sample = False

para.filter_bad_targets = False
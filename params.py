import numpy as np
from easydict import EasyDict

para = EasyDict()

para.target_mean = np.array(
    [0.814, 0.056, 0.202, 0.188, 0.559, 1.499], dtype=np.float32)
para.target_std_dev = np.array(
    [0.314, 0.486, 1.308, 0.797, 0.171, 0.316], dtype=np.float32)

para.grid_size = 0.1
para.ratio = 4

para.L1 = -40.0
para.L2 = 40.0
para.W1 = 0.0
para.W2 = 70.0
para.H1 = -2.5
para.H2 = 1.0
para.input_shape = (800, 700)
para.label_shape = (200, 175)

para.object_list = ['Car', 'Truck', 'Van']
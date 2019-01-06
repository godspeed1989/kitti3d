'''
Load pointcloud/labels from the KITTI dataset folder
'''
import os.path
import fire
import numpy as np
import time
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.spatial import Delaunay

from params import para
from utils import plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric
from kitti import (read_label_obj, read_calib_file, compute_box_3d,
                   corner_to_center_box3d, point_transform)
from pointcloud2RGB import makeBVFeature

KITTI_PATH = '/mine/KITTI_DAT'

def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (n,3) '''
    box3d_roi_inds = in_hull(pc[:,:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_fov(pc, fov, X_MIN, X_MAX, Z_MIN, Z_MAX):
    y = X_MAX * np.tan(fov * np.pi / 180)
    bbox = [[X_MIN,  0, Z_MIN], [X_MIN,  0, Z_MAX],
            [X_MAX,  y, Z_MAX], [X_MAX,  y, Z_MIN],
            [X_MAX, -y, Z_MIN], [X_MAX, -y, Z_MAX]]
    points, inds = extract_pc_in_box3d(pc, bbox)
    return points, inds

class KITTI(Dataset):
    def __init__(self, input_channels=36, frame_range=10000, use_npy=False, train=True, aug_data=False):
        self.frame_range = frame_range
        self.velo = []
        self.use_npy = use_npy
        self.aug_data = aug_data

        self.image_sets = self.load_imageset(train) # names
        # network input channels, 36 for PIXOR, 3 for RGB
        # 36 = 3.5/0.1 + 1
        self.input_channels = input_channels
        self.geometry = {
            'L1': para.L1,
            'L2': para.L2,
            'W1': para.W1,
            'W2': para.W2,
            'H1': para.H1,
            'H2': para.H2,
            'input_shape': (*para.input_shape, self.input_channels),
            'label_shape': (*para.label_shape, 1+para.box_code_len),
            'grid_size': para.grid_size,
            'ratio': para.ratio,
            'fov': 50,  # field of view in degree
        }
        self.target_mean = para.target_mean
        self.target_std_dev = para.target_std_dev

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, item):
        raw_scan = self.load_velo_scan(item)
        raw_boxes_3d_corners, raw_labelmap_boxes_3d_corners = self.get_label(item)
        if self.aug_data:
            scan, boxes_3d_corners, labelmap_boxes_3d_corners = \
                self.augment_data(raw_scan, raw_boxes_3d_corners, raw_labelmap_boxes_3d_corners)
        else:
            scan, boxes_3d_corners, labelmap_boxes_3d_corners = \
                raw_scan, raw_boxes_3d_corners, raw_labelmap_boxes_3d_corners
        if not self.use_npy:
            if self.input_channels == 36:
                scan = self.lidar_preprocess(scan)
            elif self.input_channels == 3:
                scan = self.lidar_preprocess_rgb(scan)
            elif self.input_channels == 39:
                scan1 = self.lidar_preprocess_rgb(scan)
                scan2 = self.lidar_preprocess(scan)
                scan = np.concatenate([scan1, scan2], axis=2)
        scan = torch.from_numpy(scan)
        #
        label_map, _ = self.get_label_map(boxes_3d_corners, labelmap_boxes_3d_corners)
        self.reg_target_transform(label_map)
        label_map = torch.from_numpy(label_map)
        return scan, label_map


    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std_dev


    def load_imageset(self, train):
        path = KITTI_PATH
        if train:
            path = os.path.join(path, "train.txt")
        else:
            path = os.path.join(path, "val.txt")

        with open(path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            names = []
            for line in lines[:-1]:
                if int(line[:-1]) < self.frame_range:
                    names.append(line[:-1])
            # Last line does not have a \n symbol
            names.append(lines[-1][:6])
        print("There are {} images in txt file".format(len(names)))
        return names

    def update_label_map(self, map, bev_corners, reg_target):
        label_corners = bev_corners / self.geometry['grid_size'] / self.geometry['ratio']
        # y to positive
        # XY in LiDAR <--> YX in label map
        label_corners[:, 1] += self.geometry['label_shape'][0] / 2.0

        points = get_points_in_a_rotated_box(label_corners)

        for p in points:
            metric_x, metric_y = trasform_label2metric(np.array(p),
                ratio=self.geometry['ratio'], grid_size=self.geometry['grid_size'],
                base_height=self.geometry['label_shape'][0] // 2)
            actual_reg_target = np.copy(reg_target)
            if para.box_code_len == 6:
                actual_reg_target[2] = reg_target[2] - metric_x
                actual_reg_target[3] = reg_target[3] - metric_y
                actual_reg_target[4] = np.log(reg_target[4])
                actual_reg_target[5] = np.log(reg_target[5])
            elif para.box_code_len == 5:
                actual_reg_target[1] = reg_target[1] - metric_x
                actual_reg_target[2] = reg_target[2] - metric_y
                actual_reg_target[3] = np.log(reg_target[3])
                actual_reg_target[4] = np.log(reg_target[4])
            else:
                raise NotImplementedError

            label_x = p[0]
            label_y = p[1]
            map[label_y, label_x, 0] = 1.0
            map[label_y, label_x, 1:1+para.box_code_len] = actual_reg_target

    def get_label(self, index):
        '''
        :param i: the ith velodyne scan in the train/val set
        '''
        index = self.image_sets[index]
        f_name = index + '.txt'
        label_path = os.path.join(KITTI_PATH, 'training', 'label_2', f_name)
        calib_path = os.path.join(KITTI_PATH, 'training', 'calib', f_name)

        objs = read_label_obj(label_path)
        calib_dict = read_calib_file(calib_path)

        boxes3d_corners = []
        labelmap_boxes3d_corners = []
        for obj in objs:
            if obj.type in para.object_list:
                # use calibration to get accurate position (8, 3)
                box3d_corners = compute_box_3d(obj,
                    calib_dict['R0_rect'].reshape([3,3]),
                    calib_dict['Tr_velo_to_cam'].reshape([3,4]))
                boxes3d_corners.append(box3d_corners)
                bev_obj = deepcopy(obj)
                bev_obj.w *= para.box_in_labelmap_ratio
                bev_obj.l *= para.box_in_labelmap_ratio
                bev_obj.h *= para.box_in_labelmap_ratio
                labelmap_box3d_corners = compute_box_3d(bev_obj,
                    calib_dict['R0_rect'].reshape([3,3]),
                    calib_dict['Tr_velo_to_cam'].reshape([3,4]))
                labelmap_boxes3d_corners.append(labelmap_box3d_corners)
        return boxes3d_corners, labelmap_boxes3d_corners

    def get_reg_targets(self, box3d_pts_3d, labelmap_box3d_pts_3d):
        bev_corners = box3d_pts_3d[:4, :2]
        labelmap_bev_corners = labelmap_box3d_pts_3d[:4, :2]
        #
        centers = corner_to_center_box3d(box3d_pts_3d)
        x = centers[0]
        y = centers[1]
        l = centers[3]
        w = centers[4]
        yaw = centers[6]
        if para.box_code_len == 6:
            reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
        elif para.box_code_len == 5:
            reg_target = [yaw, x, y, w, l]
        else:
            raise NotImplementedError

        return bev_corners, labelmap_bev_corners, reg_target

    def get_label_map(self, boxes3d_corners, labelmap_boxes3d_corners):
        '''return
        label map: <--- This is the learning target
            a tensor of shape 800 * 700 * 7 representing the expected output
        label_list: <--- Intended for evaluation metrics & visualization
            a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
            each entry is another list, where the first element of this list indicates if the object
            is a car or one of the 'dontcare' (truck,van,etc) object
        '''
        label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
        label_list = []
        for box3d_corners, labelmap_box3d_corners in zip(boxes3d_corners, labelmap_boxes3d_corners):
            bev_corners, labelmap_bev_corners, reg_target = self.get_reg_targets(box3d_corners, labelmap_box3d_corners)
            self.update_label_map(label_map, labelmap_bev_corners, reg_target)
            label_list.append(bev_corners)

        return label_map, label_list

    def load_velo_scan(self, item):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = self.velo[item]
        if self.use_npy:
            scan = np.load(filename[:-4]+'.npy')
        else:
            scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return scan

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_files = []
        for file in self.image_sets:
            file = '{}.bin'.format(file)
            velo_files.append(os.path.join(KITTI_PATH, 'training', 'velodyne', file))
        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')
        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = velo_files
        print('done.')

    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def lidar_preprocess(self, scan):
        # select by FOV
        if True:
            pc, ind = extract_pc_in_fov(scan[:, :3], self.geometry['fov'],
                                        self.geometry['W1'], self.geometry['W2'],
                                        self.geometry['H1'], self.geometry['H2'])
            inte = scan[ind, 3:]
            velo = np.concatenate((pc, inte), axis=1)
        else:
            velo = scan
        # generate intensity map
        velo_processed = np.zeros((*self.geometry['input_shape'][:2], 36), dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        for i in range(velo.shape[0]):
            if self.point_in_roi(velo[i, :]):
                x = int((velo[i, 1]-self.geometry['L1']) / self.geometry['grid_size'])
                y = int((velo[i, 0]-self.geometry['W1']) / self.geometry['grid_size'])
                z = int((velo[i, 2]-self.geometry['H1']) / self.geometry['grid_size'])
                velo_processed[x, y, z] = 1
                velo_processed[x, y, -1] += velo[i, 3]
                intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count, \
                        where=intensity_map_count!=0)
        return velo_processed

    def lidar_preprocess_rgb(self, scan):
        # select by FOV
        if True:
            pc, ind = extract_pc_in_fov(scan[:, :3], self.geometry['fov'],
                                        self.geometry['W1'], self.geometry['W2'],
                                        self.geometry['H1'], self.geometry['H2'])
            inte = scan[ind, 3:]
            velo = np.concatenate((pc, inte), axis=1)
        else:
            velo = scan

        size_ROI={}
        size_ROI['minX'] = self.geometry['W1']; size_ROI['maxX'] = self.geometry['W2']
        size_ROI['minY'] = self.geometry['L1']; size_ROI['maxY'] = self.geometry['L2']
        size_ROI['minZ'] = self.geometry['H1']; size_ROI['maxZ'] = self.geometry['H2']
        size_ROI['Height'] = self.geometry['input_shape'][1]
        size_ROI['Width'] = self.geometry['input_shape'][0]
        size_cell = self.geometry['grid_size']

        RGB_Map = makeBVFeature(velo, size_ROI, size_cell)

        return RGB_Map

    def augment_data(self, scan, boxes_3d_corners, labelmap_boxes_3d_corners):
        assert len(boxes_3d_corners) == len(labelmap_boxes_3d_corners)
        if len(boxes_3d_corners) > 0:
            all_corners = np.concatenate(boxes_3d_corners, axis=0)
            labelmap_corners = np.concatenate(labelmap_boxes_3d_corners, axis=0)
        else:
            all_corners = None
            labelmap_corners = None
        if np.random.choice(2):
            # global rotation
            angle = np.random.uniform(-np.pi / 8, np.pi / 8)
            scan[:, 0:3] = point_transform(scan[:, 0:3], 0, 0, 0, rz=angle)
            if all_corners is not None:
                all_corners = point_transform(all_corners, 0, 0, 0, rz=angle)
                labelmap_corners = point_transform(labelmap_corners, 0, 0, 0, rz=angle)
        if np.random.choice(2):
            # global translation
            tx = np.random.uniform(-1, 1)
            ty = np.random.uniform(-1, 1)
            tz = np.random.uniform(-0.15, 0.15)
            scan[:, 0:3] = point_transform(scan[:, 0:3], tx, ty, tz)
            if all_corners is not None:
                all_corners = point_transform(all_corners, tx, ty, tz)
                labelmap_corners = point_transform(labelmap_corners, tx, ty, tz)
        if np.random.choice(2):
            # global scaling
            factor = np.random.uniform(0.9, 1.1)
            scan[:, 0:3] = scan[:, 0:3] * factor
            if all_corners is not None:
                all_corners = all_corners * factor
                labelmap_corners = labelmap_corners * factor
        ret_boxes_3d_corners = []
        ret_labelmap_boxes_3d_corners = []
        for i in range(len(boxes_3d_corners)):
            ret_boxes_3d_corners.append(all_corners[i*8 : (i+1)*8])
            ret_labelmap_boxes_3d_corners.append(labelmap_corners[i*8 : (i+1)*8])
        return scan, ret_boxes_3d_corners, ret_labelmap_boxes_3d_corners

def get_data_loader(batch_size=4, input_channels=36, use_npy=False, frame_range=10000, workers=4):
    train_dataset = KITTI(frame_range=frame_range, input_channels=input_channels, use_npy=use_npy, train=True, aug_data=True)
    train_dataset.load_velo()
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=workers)

    val_dataset = KITTI(frame_range=frame_range, input_channels=input_channels, use_npy=use_npy, train=False, aug_data=False)
    val_dataset.load_velo()
    val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=workers)

    return train_data_loader, val_data_loader

#########################################################################################

def test0():
    k = KITTI()
    id = 2
    k.load_velo()
    tstart = time.time()
    scan = k.load_velo_scan(id)
    boxes_3d_corners, labelmap_boxes_3d_corners = k.get_label(id)
    scan, boxes_3d_corners, labelmap_boxes_3d_corners = k.augment_data(scan, boxes_3d_corners, labelmap_boxes_3d_corners)
    processed_v = k.lidar_preprocess(scan)
    label_map, label_list = k.get_label_map(boxes_3d_corners, labelmap_boxes_3d_corners)
    print('time taken: %gs' %(time.time()-tstart))
    plot_bev(processed_v, label_list)
    plot_label_map(label_map[:, :, :3])

def find_reg_target_var_and_mean():
    k = KITTI(train=True)
    k.load_velo()
    reg_targets = [[] for _ in range(para.box_code_len)]
    for i in range(len(k)):
        boxes_3d_corners, labelmap_boxes_3d_corners = k.get_label(i)
        label_map, _ = k.get_label_map(boxes_3d_corners, labelmap_boxes_3d_corners)
        car_locs = np.where(label_map[:, :, 0] == 1)
        for j in range(1, 1+para.box_code_len):
            map = label_map[:, :, j]
            reg_targets[j-1].extend(list(map[car_locs]))

    reg_targets = np.array(reg_targets)
    means = reg_targets.mean(axis=1)
    stds = reg_targets.std(axis=1)

    np.set_printoptions(precision=3, suppress=True)
    print("Means", means)
    print("Stds", stds)
    return means, stds

def preprocess_to_npy(train=True):
    k = KITTI(train=train)
    k.load_velo()
    for item, name in enumerate(k.velo):
        scan = k.load_velo_scan(item)
        scan = k.lidar_preprocess(scan)
        path = name[:-4] + '.npy'
        np.save(path, scan)
        print('Saved ', path)
    return

def test():
    train_data_loader, val_data_loader = get_data_loader(batch_size=2)
    for i, (input, label_map) in enumerate(train_data_loader):
        print("Entry", i)
        print("Input shape:", input.shape)
        print("Label Map shape", label_map.shape)
        if i == 5:
            break

    print("Finish testing train dataloader")


if __name__=="__main__":
    fire.Fire()

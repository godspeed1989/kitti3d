'''
Load pointcloud/labels from the KITTI dataset folder
'''
import os.path
import cv2
import fire
import numpy as np
import time
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.spatial import Delaunay
from collections import defaultdict

from params import para
from utils import (plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric,
                   remove_points_in_boxes)
from kitti import (read_label_obj, read_calib_file, compute_lidar_box_3d, lidar_center_to_corner_box3d,
                   corner_to_center_box3d, point_transform, angle_in_limit)
from gt_db_sampler import DataBaseSampler, fake_boxes_corners3d
from pointcloud2RGB import makeBVFeature
from voxel_gen import VoxelGenerator

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
    def __init__(self, frame_range=10000, selection='train', aug_data=False):
        self.frame_range = frame_range
        self.velo = []
        self.aug_data = aug_data
        self.align_pc_with_img = para.align_pc_with_img
        self.crop_pc_by_fov = para.crop_pc_by_fov

        assert selection in ['train', 'val', 'trainval', 'test']
        self.selection = selection
        if self.selection == 'test':
            self.data_dir = 'testing'
        else:
            self.data_dir = 'training'
        self.image_sets = self.load_imageset()
        self.input_channels = para.input_channels
        self.geometry = {
            'L1': para.L1,
            'L2': para.L2,
            'W1': para.W1,
            'W2': para.W2,
            'H1': para.H1,
            'H2': para.H2,
            'input_shape': (*para.input_shape, self.input_channels),
            'label_shape': (*para.label_shape, 1+para.box_code_len),
            'ratio': para.ratio,
            'fov': 50,  # field of view in degree
        }
        self.target_mean = para.target_mean
        self.target_std_dev = para.target_std_dev
        if para.augment_data_use_db:
            self.sampler = DataBaseSampler(KITTI_PATH, os.path.join(KITTI_PATH, "kitti_dbinfos_train.pkl"))
        self.voxel_generator = VoxelGenerator(
            voxel_size=[para.grid_sizeLW, para.grid_sizeLW, para.grid_sizeH],
            point_cloud_range=[para.W1, para.L1, para.H1,
                               para.W2, para.L2, para.H2],
            max_num_points=30,
            max_voxels=40000
        )

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, item):
        ret = {}
        raw_scan = self.load_velo_scan(item)

        if self.selection == 'test':
            assert self.aug_data == False
            index, calib_dict = self.get_label(item)
            raw_boxes_3d_corners, raw_labelmap_boxes_3d_corners, \
                raw_labelmap_mask_boxes_3d_corners = None, None, None
        else:
            index, raw_boxes_3d_corners, raw_labelmap_boxes_3d_corners, \
                raw_labelmap_mask_boxes_3d_corners, calib_dict = self.get_label(item)

        if self.align_pc_with_img:
            raw_scan = self.align_img_and_pc(raw_scan, calib_dict)

        if self.crop_pc_by_fov:
            raw_scan = self.crop_pc_using_fov(raw_scan)

        if self.aug_data:
            scan, boxes_3d_corners, labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners = \
                self.augment_data(index, raw_scan, raw_boxes_3d_corners, raw_labelmap_boxes_3d_corners, raw_labelmap_mask_boxes_3d_corners)
        else:
            scan, boxes_3d_corners, labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners = \
                raw_scan, raw_boxes_3d_corners, raw_labelmap_boxes_3d_corners, raw_labelmap_mask_boxes_3d_corners

        if para.channel_type == 'rgb':
            ret['scan'] = self.lidar_preprocess_rgb(scan)
        if para.channel_type == 'pixor':
            ret['scan'] = self.lidar_preprocess(scan)
        if para.channel_type == 'pixor-rgb':
            scan1 = self.lidar_preprocess_rgb(scan)
            scan2 = self.lidar_preprocess(scan)
            ret['scan'] = np.concatenate([scan1, scan2], axis=2)
        if para.channel_type == 'voxel':
            ret['scan'] = self.lidar_preprocess_voxel(scan)
        if para.channel_type == 'sparse':
            ret['voxels_feature'], ret['coordinates'] = self.lidar_preprocess_sparse(scan)

        if self.selection != 'test':
            label_map, _, label_map_mask = self.get_label_map(boxes_3d_corners, \
                labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners)
            self.reg_target_transform(label_map)
            ret['label_map'] = label_map
            ret['label_map_mask'] = label_map_mask

        return ret

    def crop_pc_using_fov(self, raw_scan):
        pc, ind = extract_pc_in_fov(raw_scan[:, :3], self.geometry['fov'],
                                    self.geometry['W1'], self.geometry['W2'],
                                    self.geometry['H1'], self.geometry['H2'])
        inte = raw_scan[ind, 3:]
        raw_scan = np.concatenate((pc, inte), axis=1)
        return raw_scan

    def align_img_and_pc(self, raw_scan, calib_dict):
        P = calib_dict['P2'].reshape([3,4])
        Tr_velo_to_cam = calib_dict['Tr_velo_to_cam'].reshape([3,4])
        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)], axis=0)
        R_cam_to_rect = np.eye(4)
        R_cam_to_rect[:3,:3] = calib_dict['R0_rect'].reshape(3,3)

        def prepare_velo_points(pts3d_raw):
            '''Replaces the reflectance value by 1, and tranposes the array, so
                points can be directly multiplied by the camera projection matrix'''
            pts3d = pts3d_raw
            # Reflectance > 0
            indices = pts3d[:, 3] >= 0
            pts3d = pts3d[indices,:]
            pts3d[:,3] = 1
            return pts3d.transpose(), indices
        pts3d, indices = prepare_velo_points(raw_scan)

        reflectances = raw_scan[indices, 3]
        def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
            '''Project 3D points into 2D image. Expects pts3d as a 4xN numpy array.
            Returns the 3D and 2D projection of the points are in front of the camera
            '''
            # 3D points in camera reference frame.
            pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
            # keep only points with z>0
            # (points that are in front of the camera).
            idx = (pts3d_cam[2, :]>=0)
            pts2d_cam = Prect.dot(pts3d_cam[:, idx])
            return pts3d[:, idx], pts2d_cam/pts2d_cam[2,:], idx
        pts3d, pts2d_normed, idx = project_velo_points_in_img(pts3d, Tr_velo_to_cam, R_cam_to_rect, P)
        reflectances = reflectances[idx]
        assert reflectances.shape[0] == pts2d_normed.shape[1] == pts2d_normed.shape[1]

        rows, cols = para.img_shape

        points = []
        for i in range(pts2d_normed.shape[1]):
            c = int(np.round(pts2d_normed[0,i]))
            r = int(np.round(pts2d_normed[1,i]))
            if c < cols and r < rows and r > 0 and c > 0:
                point = [ pts3d[0,i], pts3d[1,i], pts3d[2,i], reflectances[i] ]
                points.append(point)

        points = np.array(points)
        return points

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


    def load_imageset(self):
        path = KITTI_PATH
        if self.selection == 'train':
            path = os.path.join(path, "train.txt")
        elif self.selection == 'val':
            path = os.path.join(path, "val.txt")
        elif self.selection == 'trainval':
            path = os.path.join(path, "trainval.txt")
        elif self.selection == 'test':
            path = os.path.join(path, "test.txt")
        else:
            raise NotImplementedError

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

    def update_label_map(self, labelmap, bev_corners, map_mask, bev_corners_mask, reg_target):
        label_corners = bev_corners / para.grid_sizeLW / self.geometry['ratio']
        # y to positive
        # XY in LiDAR <--> YX in label map
        label_corners[:, 1] += self.geometry['label_shape'][0] / 2.0

        points = get_points_in_a_rotated_box(label_corners)

        for p in points:
            metric_x, metric_y = trasform_label2metric(np.array(p),
                ratio=self.geometry['ratio'], grid_size=para.grid_sizeLW,
                base_height=self.geometry['label_shape'][0] // 2)
            actual_reg_target = np.copy(reg_target)
            if para.box_code_len == 6 or para.box_code_len == 8:
                actual_reg_target[2] = reg_target[2] - metric_x
                actual_reg_target[3] = reg_target[3] - metric_y
                actual_reg_target[4] = np.log(reg_target[4])
                actual_reg_target[5] = np.log(reg_target[5])
            elif para.box_code_len == 5 or para.box_code_len == 7:
                actual_reg_target[1] = reg_target[1] - metric_x
                actual_reg_target[2] = reg_target[2] - metric_y
                actual_reg_target[3] = np.log(reg_target[3])
                actual_reg_target[4] = np.log(reg_target[4])
                if para.sin_angle_loss:
                    actual_reg_target[0] = np.sin(reg_target[0])
            else:
                raise NotImplementedError
            if para.estimate_bh:
                actual_reg_target[-2] = np.log(reg_target[-2] + para.height_bias)
                actual_reg_target[-1] = np.log(reg_target[-1] + para.height_bias)

            label_x = p[0]
            label_y = p[1]
            labelmap[label_y, label_x, 0] = 1.0
            labelmap[label_y, label_x, 1:1+para.box_code_len] = actual_reg_target
        #
        label_corners_mask = bev_corners_mask / para.grid_sizeLW / self.geometry['ratio']
        label_corners_mask[:, 1] += self.geometry['label_shape'][0] / 2.0
        points_mask = get_points_in_a_rotated_box(label_corners_mask)
        for p in points_mask:
            label_x = p[0]
            label_y = p[1]
            map_mask[label_y, label_x] = 0.5

    def get_label(self, idx):
        '''
        :param i: the ith velodyne scan in the train/val set
        '''
        index = self.image_sets[idx]
        f_name = index + '.txt'
        label_path = os.path.join(KITTI_PATH, self.data_dir, 'label_2', f_name)
        calib_path = os.path.join(KITTI_PATH, self.data_dir, 'calib', f_name)

        calib_dict = read_calib_file(calib_path)
        if self.selection == 'test':
            return index, calib_dict

        def obj_good_to_train(obj):
            if para.filter_bad_targets:
                return np.sqrt(obj.t[0]**2 + obj.t[2]**2) < 50
            else:
                return True

        objs = read_label_obj(label_path)
        boxes3d_corners = []
        labelmap_boxes3d_corners = []
        labelmap_mask_boxes3d_corners = []
        for obj in objs:
            if obj.type in para.object_list and obj_good_to_train(obj):
                # use calibration to get accurate position (8, 3)
                box3d_corners = compute_lidar_box_3d(obj,
                    calib_dict['R0_rect'].reshape([3,3]),
                    calib_dict['Tr_velo_to_cam'].reshape([3,4]))
                boxes3d_corners.append(box3d_corners)
                #
                bev_obj = deepcopy(obj)
                bev_obj.w *= para.box_in_labelmap_ratio
                bev_obj.l *= para.box_in_labelmap_ratio
                bev_obj.h *= para.box_in_labelmap_ratio
                labelmap_box3d_corners = compute_lidar_box_3d(bev_obj,
                    calib_dict['R0_rect'].reshape([3,3]),
                    calib_dict['Tr_velo_to_cam'].reshape([3,4]))
                labelmap_boxes3d_corners.append(labelmap_box3d_corners)
                #
                mask_obj = deepcopy(obj)
                mask_obj.w *= para.box_in_labelmap_mask_ratio
                mask_obj.l *= para.box_in_labelmap_mask_ratio
                mask_obj.h *= para.box_in_labelmap_mask_ratio
                labelmap_mask_box3d_corners = compute_lidar_box_3d(mask_obj,
                    calib_dict['R0_rect'].reshape([3,3]),
                    calib_dict['Tr_velo_to_cam'].reshape([3,4]))
                labelmap_mask_boxes3d_corners.append(labelmap_mask_box3d_corners)
        return index, boxes3d_corners, labelmap_boxes3d_corners, labelmap_mask_boxes3d_corners, calib_dict

    def get_reg_targets(self, box3d_pts_3d):
        bev_corners = box3d_pts_3d[:4, :2]
        head = np.max(box3d_pts_3d[:, 2])
        bottom = np.min(box3d_pts_3d[:, 2])
        #
        centers = corner_to_center_box3d(box3d_pts_3d)
        x = centers[0]
        y = centers[1]
        l = centers[3]
        w = centers[4]
        yaw = angle_in_limit(-centers[6])
        if para.box_code_len == 6:
            reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
        elif para.box_code_len == 5:
            reg_target = [yaw, x, y, w, l]
        elif para.box_code_len == 8:
            reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l, bottom, head]
        elif para.box_code_len == 7:
            reg_target = [yaw, x, y, w, l, bottom, head]
        else:
            raise NotImplementedError

        return bev_corners, reg_target

    def get_label_map(self, boxes3d_corners, labelmap_boxes3d_corners, labelmap_mask_boxes3d_corners):
        '''return
        label map: <--- This is the learning target
            a tensor of shape 800 * 700 * 7 representing the expected output
        label_list: <--- Intended for evaluation metrics & visualization
            a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
            each entry is another list, where the first element of this list indicates if the object
            is a car or one of the 'dontcare' (truck,van,etc) object
        '''
        label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
        label_map_mask = np.ones(para.label_shape, dtype=np.float32)
        label_list = []
        for box3d_corners, labelmap_box3d_corners, labelmap_mask_box3d_corners in \
                zip(boxes3d_corners, labelmap_boxes3d_corners, labelmap_mask_boxes3d_corners):
            bev_corners, reg_target = self.get_reg_targets(box3d_corners)
            labelmap_bev_corners = labelmap_box3d_corners[:4, :2]
            labelmap_mask_bev_corners = labelmap_mask_box3d_corners[:4, :2]
            self.update_label_map(label_map, labelmap_bev_corners,
                label_map_mask, labelmap_mask_bev_corners, reg_target)
            label_list.append(bev_corners)
        label_map_mask = label_map_mask + label_map[:,:,0] * 0.5

        return label_map, label_list, label_map_mask

    def load_velo_scan(self, item):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = self.velo[item]
        scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return scan

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_files = []
        for file in self.image_sets:
            file = '{}.bin'.format(file)
            velo_files.append(os.path.join(KITTI_PATH, self.data_dir, 'velodyne', file))
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

    def lidar_preprocess(self, velo):
        # generate intensity map
        channels = int((para.H2 - para.H1) / para.grid_sizeH + 1)
        velo_processed = np.zeros((*self.geometry['input_shape'][:2], channels), dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        for i in range(velo.shape[0]):
            if self.point_in_roi(velo[i, :]):
                x = int((velo[i, 1]-self.geometry['L1']) / para.grid_sizeLW)
                y = int((velo[i, 0]-self.geometry['W1']) / para.grid_sizeLW)
                z = int((velo[i, 2]-self.geometry['H1']) / para.grid_sizeH)
                velo_processed[x, y, z] = 1
                velo_processed[x, y, -1] += velo[i, 3]
                intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count, \
                        where=intensity_map_count!=0)
        return velo_processed

    def lidar_preprocess_rgb(self, velo):
        size_ROI={}
        size_ROI['minX'] = self.geometry['W1']; size_ROI['maxX'] = self.geometry['W2']
        size_ROI['minY'] = self.geometry['L1']; size_ROI['maxY'] = self.geometry['L2']
        size_ROI['minZ'] = self.geometry['H1']; size_ROI['maxZ'] = self.geometry['H2']
        size_ROI['Height'] = self.geometry['input_shape'][1]
        size_ROI['Width'] = self.geometry['input_shape'][0]
        size_cell = para.grid_sizeLW

        RGB_Map = makeBVFeature(velo, size_ROI, size_cell)

        return RGB_Map

    def lidar_preprocess_voxel(self, velo):
        channels = int((para.H2 - para.H1) / para.grid_sizeH)
        velo_processed = np.zeros((*self.geometry['input_shape'][:2], channels), dtype=np.float32)
        # X,Y,Z  ->  Z,Y,X
        voxels, coords, num_points_per_voxel = self.voxel_generator.generate(velo.astype(np.float32))
        for i,p in enumerate(coords):
            inte = np.sum(voxels[i,:,-1]) / num_points_per_voxel[i]
            velo_processed[p[1],p[2],p[0]] = inte
        return velo_processed

    def lidar_preprocess_sparse(self, velo):
        # X,Y,Z  ->  Z,Y,X
        voxels, coords, num_points_per_voxel = self.voxel_generator.generate(velo.astype(np.float32))
        voxels = voxels.astype(np.float32)      # (M, K, 4)
        coords = coords.astype(np.int32)        # (M, 3)
        num_points_per_voxel = num_points_per_voxel.astype(np.int32)    # (M,)
        # (M, C)
        if para.voxel_feature_len == 2:
            voxels_feature = voxels[:, :, 2:4].sum(axis=1, keepdims=False)
        elif para.voxel_feature_len == 4:
            voxels_feature = voxels[:, :, :4].sum(axis=1, keepdims=False)
        voxels_feature = voxels_feature / num_points_per_voxel.astype(voxels.dtype)[..., np.newaxis]
        return voxels_feature, coords

    def augment_data(self, index, scan, boxes_3d_corners,
                     labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners):
        assert len(boxes_3d_corners) == len(labelmap_boxes_3d_corners)
        assert len(boxes_3d_corners) == len(labelmap_mask_boxes_3d_corners)
        if len(boxes_3d_corners) > 0:
            # [(8,3),...,(8,3)] -> (N,8,3)
            all_corners = np.stack(boxes_3d_corners)
            labelmap_corners = np.stack(labelmap_boxes_3d_corners)
            labelmap_mask_corners = np.stack(labelmap_mask_boxes_3d_corners)
        else:
            all_corners = np.zeros([0,8,3])
            labelmap_corners = np.zeros([0,8,3])
            labelmap_mask_corners = np.zeros([0,8,3])

        if para.augment_data_use_db:
            collision_corners = all_corners
            sampled = self.sampler.sample_all('Car', collision_corners, para.augment_max_samples)
            # ----
            def filter_by_ground(sampled, index):
                if sampled is None:
                    return None
                centers3d = sampled["boxes_centers3d"]
                corners2d = lidar_center_to_corner_box3d(centers3d)[:,:4,:2]
                def line_to_poly(line):
                    ret = []
                    for s in line.split(' '):
                        if s.isnumeric():
                            ret.append(int(s))
                    ret = np.array(ret, dtype=np.int32)
                    return np.reshape(ret, [-1, 2])
                def load_anno(idx):
                    ANNO_DST = os.path.join(KITTI_PATH, 'training/grd_mask')
                    filename = os.path.join(ANNO_DST, index+'.txt')
                    ret = []
                    if os.path.exists(filename):
                        with open(filename, 'r') as f:
                            lines = f.readlines()
                            for l in lines:
                                pl = line_to_poly(l)
                                ret.append(pl)
                            f.close()
                    return ret
                grd_poly = load_anno(index)
                if len(grd_poly) < 1:
                    return None
                mask = np.zeros(self.geometry['input_shape'][:2], np.uint8)
                cv2.fillPoly(mask, grd_poly, 255)
                #
                names = []
                gt_boxes = []
                points_list = []
                for i in range(corners2d.shape[0]):
                    cnt = 0
                    for j in range(4):
                        x = int((corners2d[i,j,1]-self.geometry['L1']) / para.grid_sizeLW)
                        y = int((corners2d[i,j,0]-self.geometry['W1']) / para.grid_sizeLW)
                        if not(x>=0 and x<self.geometry['input_shape'][0] and \
                                y>=0 and y<self.geometry['input_shape'][1]):
                            break
                        if mask[x,y] < 1:
                            break
                        cnt += 1
                    if cnt == 4:
                        names.append(sampled['names'][i])
                        gt_boxes.append(sampled['boxes_centers3d'][i])
                        points_list.append(sampled['points'][i])
                if len(names) > 0:
                    return {
                        "names": names,
                        "boxes_centers3d": np.stack(gt_boxes, axis=0),
                        "points": points_list,
                    }
                else:
                    return None
            # ----
            if para.filter_sampled_by_ground:
                sampled = filter_by_ground(sampled, index)
            if sampled is not None:
                sampled_boxes_centers3d = sampled["boxes_centers3d"].copy()
                # gt
                sampled_boxes_corners3d = lidar_center_to_corner_box3d(sampled_boxes_centers3d)
                all_corners = np.concatenate([all_corners, sampled_boxes_corners3d], axis=0)
                # labelmap
                labelmap_sampled_boxes_centers3d = sampled_boxes_centers3d.copy()
                labelmap_sampled_boxes_centers3d[:, 3:6] *= para.box_in_labelmap_ratio
                labelmap_sampled_boxes_corners3d = lidar_center_to_corner_box3d(labelmap_sampled_boxes_centers3d)
                labelmap_corners = np.concatenate([labelmap_corners, labelmap_sampled_boxes_corners3d], axis=0)
                # labelmap mask
                labelmap_mask_sampled_boxes_centers3d = sampled_boxes_centers3d.copy()
                labelmap_mask_sampled_boxes_centers3d[:, 3:6] *= para.box_in_labelmap_mask_ratio
                labelmap_mask_sampled_boxes_corners3d = lidar_center_to_corner_box3d(labelmap_mask_sampled_boxes_centers3d)
                labelmap_mask_corners = np.concatenate([labelmap_mask_corners, labelmap_mask_sampled_boxes_corners3d], axis=0)
                # remove
                if para.remove_points_after_sample:
                    boxes_corners3d = sampled_boxes_corners3d.copy()
                    # just to call points_in_convex_polygon_3d_jit correct
                    boxes_corners3d = boxes_corners3d[:,(0,3,2,1,4,7,6,5),:]
                    boxes_corners3d[:,:4,2] = -10
                    boxes_corners3d[:,4:,2] = 10
                    scan, _ = remove_points_in_boxes(scan, boxes_corners3d)
                scan = np.vstack([scan, np.concatenate(sampled["points"],axis=0)])

        num_target = all_corners.shape[0]
        # (N,8,3) -> (N*8,3)
        all_corners = np.reshape(all_corners, [num_target*8, 3])
        labelmap_corners = np.reshape(labelmap_corners, [num_target*8, 3])
        labelmap_mask_corners = np.reshape(labelmap_mask_corners, [num_target*8, 3])

        if np.random.choice(2):
            # global rotation
            angle = np.random.uniform(-np.pi / 8, np.pi / 8)
            scan[:, 0:3] = point_transform(scan[:, 0:3], 0, 0, 0, rz=angle)
            if num_target > 0:
                all_corners = point_transform(all_corners, 0, 0, 0, rz=angle)
                labelmap_corners = point_transform(labelmap_corners, 0, 0, 0, rz=angle)
                labelmap_mask_corners = point_transform(labelmap_mask_corners, 0, 0, 0, rz=angle)
        if np.random.choice(2):
            # global translation
            tx = np.random.uniform(-1, 1)
            ty = np.random.uniform(-1, 1)
            tz = np.random.uniform(-0.15, 0.15)
            scan[:, 0:3] = point_transform(scan[:, 0:3], tx, ty, tz)
            if num_target > 0:
                all_corners = point_transform(all_corners, tx, ty, tz)
                labelmap_corners = point_transform(labelmap_corners, tx, ty, tz)
                labelmap_mask_corners = point_transform(labelmap_mask_corners, tx, ty, tz)
        if np.random.choice(2):
            # global scaling
            factor = np.random.uniform(0.95, 1.05)
            scan[:, 0:3] = scan[:, 0:3] * factor
            if num_target > 0:
                all_corners = all_corners * factor
                labelmap_corners = labelmap_corners * factor
                labelmap_mask_corners = labelmap_mask_corners * factor
        if np.random.choice(2):
            # XY flip
            scan[:, 0] = self.geometry['W2'] - scan[:, 0] + self.geometry['W1']
            scan[:, 1] = self.geometry['H2'] - scan[:, 1] + self.geometry['H1']
            if num_target > 0:
                all_corners[:,0] = self.geometry['W2'] - all_corners[:,0] + self.geometry['W1']
                labelmap_corners[:,0] = self.geometry['W2'] - labelmap_corners[:,0] + self.geometry['W1']
                labelmap_mask_corners[:,0] = self.geometry['W2'] - labelmap_mask_corners[:,0] + self.geometry['W1']
                all_corners[:,1] = self.geometry['H2'] - all_corners[:,1] + self.geometry['H1']
                labelmap_corners[:,1] = self.geometry['H2'] - labelmap_corners[:,1] + self.geometry['H1']
                labelmap_mask_corners[:,1] = self.geometry['H2'] - labelmap_mask_corners[:,1] + self.geometry['H1']

        ret_boxes_3d_corners = []
        ret_labelmap_boxes_3d_corners = []
        ret_labelmap_mask_boxes_3d_corners = []
        for i in range(num_target):
            ret_boxes_3d_corners.append(all_corners[i*8 : (i+1)*8])
            ret_labelmap_boxes_3d_corners.append(labelmap_corners[i*8 : (i+1)*8])
            ret_labelmap_mask_boxes_3d_corners.append(labelmap_mask_corners[i*8 : (i+1)*8])
        return scan, ret_boxes_3d_corners, ret_labelmap_boxes_3d_corners, ret_labelmap_mask_boxes_3d_corners

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    #print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def _merge_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}

    for key, elems in example_merged.items():
        if key in ['voxels_feature']:
            # [N,C] + [M,C] -> [M+N,C]
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            # add batch number: [M, 3](xyz) -> [M, 4](bxyz)
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            # [A,B] + [A,B] -> [2,A,B]
            ret[key] = np.stack(elems, axis=0)
    for k,v in ret.items():
        ret[k] = torch.from_numpy(v)
    ret['cur_batch_size'] = len(batch_list)
    return ret

def get_data_loader(db_selection, batch_size=4,
                    frame_range=10000, workers=4,
                    shuffle=False, augment=False):
    dataset = KITTI(frame_range=frame_range, selection=db_selection, aug_data=augment)
    dataset.load_velo()
    data_loader = DataLoader(dataset,
            shuffle=shuffle,
            pin_memory=False,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=_merge_batch,
            worker_init_fn=_worker_init_fn)

    return data_loader

#########################################################################################

def test0():
    k = KITTI(selection='train')
    for id in range(len(k)):
        k.load_velo()
        tstart = time.time()
        scan = k.load_velo_scan(id)
        index, boxes_3d_corners, labelmap_boxes_3d_corners, \
            labelmap_mask_boxes_3d_corners, calib_dict = k.get_label(id)
        if k.align_pc_with_img:
            scan = k.align_img_and_pc(scan, calib_dict)
        if k.crop_pc_by_fov:
            scan = k.crop_pc_using_fov(scan)
        scan, boxes_3d_corners, labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners = \
            k.augment_data(index, scan, boxes_3d_corners, labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners)
        label_map, label_list, label_map_mask = \
            k.get_label_map(boxes_3d_corners, labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners)
        RGB_Map = k.lidar_preprocess_rgb(scan)
        print('time taken: %gs' %(time.time()-tstart))
        plot_bev(RGB_Map, label_list=label_list)
        plot_label_map(label_map[:, :, :3])
        plot_label_map(label_map_mask)

def find_reg_target_var_and_mean():
    k = KITTI(selection='trainval')
    k.load_velo()
    reg_targets = [[] for _ in range(para.box_code_len)]
    for i in range(len(k)):
        _, boxes_3d_corners, labelmap_boxes_3d_corners, \
            labelmap_mask_boxes_3d_corners, _ = k.get_label(i)
        label_map, _, _ = k.get_label_map(boxes_3d_corners, \
            labelmap_boxes_3d_corners, labelmap_mask_boxes_3d_corners)
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


def test():
    train_data_loader = get_data_loader('train', batch_size=2)
    for i, data in enumerate(train_data_loader):
        print("Entry", i)
        if para.dense_net:
            scan = data['scan']
            print('Scan Shape', scan.shape)
        else:
            voxels_feature = data['voxels_feature']
            coords_pad = data['coordinates']
            print('Voxel Feature Shape', voxels_feature.shape)
            print('Coords_pad Shape', coords_pad.shape)
        label_map = data['label_map']
        label_map_mask = data['label_map_mask']
        print("Label Map shape", label_map.shape)
        print("Label Map Mask shape", label_map_mask.shape)
        if i == 5:
            break

    print("Finish testing train dataloader")


if __name__=="__main__":
    fire.Fire()

import abc
import copy
import pathlib
import pickle
import numpy as np
import numba
from viewer import view_pc
from kitti import lidar_center_to_corner_box3d, corner_to_center_box3d, point_transform

root_path = "/mine/KITTI_DAT/"
database_info_path = root_path + "kitti_dbinfos_train.pkl"

class BatchSampler:
    def __init__(self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._name is not None:
            print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]
        # return np.random.choice(self._sampled_list, num)

@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result

@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(
            dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners

class DataBasePreprocessing:
    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(self, db_infos):
        pass

class DBFilterByDifficulty(DataBasePreprocessing):
    def __init__(self, removed_difficulties=[-1]):
        self._removed_difficulties = removed_difficulties
        print(removed_difficulties)

    def _preprocess(self, db_infos):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info["difficulty"] not in self._removed_difficulties
            ]
        return new_db_infos

class DBFilterByMinNumPoint(DataBasePreprocessing):
    def __init__(self, min_gt_point_dict={"Car": 10}):
        self._min_gt_point_dict = min_gt_point_dict
        print(min_gt_point_dict)

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

class DataBaseSampler:
    def __init__(self, root_path, info_path, num_point_features=4):
        self.root_path = root_path
        self.num_point_features = num_point_features
        #
        with open(info_path, 'rb') as f:
            db_infos = pickle.load(f)
        #
        prepors = [DBFilterByDifficulty(), DBFilterByMinNumPoint()]
        for prepor in prepors:
            db_infos = prepor(db_infos)
        #
        self._sampler_dict = {}
        for k, v in db_infos.items():
            self._sampler_dict[k] = BatchSampler(v, k)

    def print_class_name(self):
        print(self._sampler_dict.keys())

    def sample_n_obj(self, class_name, sampled_num):
        all_samples = self._sampler_dict[class_name].sample(sampled_num)
        samples = copy.deepcopy(all_samples)
        return samples

    def load_sample_points(self, all_samples, augment=True):
        s_points_list = []
        for info in all_samples:
            # print(info.keys())
            s_points = np.fromfile(
                str(pathlib.Path(self.root_path) / info["path"]),
                dtype=np.float32)
            s_points = s_points.reshape([-1, self.num_point_features])
            if "rot_transform" in info:
                assert 0
            # print(s_points.shape, info["num_points_in_gt"])
            s_points[:, :3] += info["box3d_lidar"][:3]
            if augment:
                clip = 0.01
                jittered_data = np.clip(0.01 * np.random.randn(s_points.shape[0], 3), -1*clip, clip)
                s_points[:, :3] += jittered_data
            s_points_list.append(s_points)
        return s_points_list

    def sample_class_v2(self, sampled, sampled_points, gt_boxes):
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        # BEV: ground truth
        gt_boxes_bv = gt_boxes[:, :4, :2]

        # BEV: sampled
        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_boxes_bv = center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        valid_points = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
                valid_points.append(sampled_points[i - num_gt])
        return valid_samples, valid_points

    def sample_all(self, class_name, gt_boxes, max_sample=15):
        '''
            Input:
                class_name 'Car'
                gt_boxes (N, 8, 3)
            Return:
                lidar centers box 3d (N', 7)
        '''
        if max_sample > gt_boxes.shape[0]:
            sample_cnt = max_sample - gt_boxes.shape[0]
        else:
            return None

        all_samples = self.sample_n_obj(class_name, sample_cnt)
        all_points = self.load_sample_points(all_samples)

        # random translation, point_transform seems bad
        for i in range(len(all_samples)):
            tx = np.random.uniform(0, 0.1)
            ty = np.random.uniform(0, 0.1)
            r_boxes_centers3d = all_samples[i]['box3d_lidar'].copy()
            r_boxes_centers3d[:2] = r_boxes_centers3d[:2] + [tx, ty]
            all_samples[i]['box3d_lidar'] = r_boxes_centers3d
            #
            r_points = all_points[i].copy()
            r_points[:, :2] = r_points[:, :2] + [tx, ty]
            all_points[i] = r_points

        # filter bad samples
        sampled_cls, points_cls = self.sample_class_v2(all_samples, all_points, gt_boxes)
        if len(sampled_cls) > 0:
            if len(sampled_cls) == 1:
                sampled_gt_boxes = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
            else:
                sampled_gt_boxes = np.stack([s["box3d_lidar"] for s in sampled_cls], axis=0)

            # TODO: just to call lidar_center_to_corner_box3d right
            sampled_gt_boxes[:,6] = -sampled_gt_boxes[:,6] - np.pi/2

            ret = {
                "names": [s["name"] for s in sampled_cls],
                "boxes_centers3d": sampled_gt_boxes,
                "points": points_cls,
            }
        else:
            ret = None

        return ret


import os
from kitti import read_calib_file, read_label_obj, compute_lidar_box_3d, reorg_calib_dict
index = '000044'
KITTI_PATH = '/mine/KITTI_DAT/training'

def _get_gt():
    f_name = index + '.txt'
    label_path = os.path.join(KITTI_PATH, 'label_2', f_name)
    calib_path = os.path.join(KITTI_PATH, 'calib', f_name)
    calib_dict = read_calib_file(calib_path)
    objs = read_label_obj(label_path)

    gt_boxes_corners3d = []
    for obj in objs:
        if obj.type in ['Car']:
            box3d_corners = compute_lidar_box_3d(obj,
                calib_dict['R0_rect'].reshape([3,3]),
                calib_dict['Tr_velo_to_cam'].reshape([3,4]))
            gt_boxes_corners3d.append(box3d_corners)
    gt_boxes_corners3d = np.stack(gt_boxes_corners3d)
    return gt_boxes_corners3d

def _get_calib_dict(idx):
    calib_path = os.path.join(KITTI_PATH, 'calib', '%s.txt' % idx)
    calib_dict = read_calib_file(calib_path)
    return calib_dict

def get_road_plane(plane_file_path):
    with open(plane_file_path, 'r') as f:
        lines = f.readlines()
    lines = [float(i) for i in lines[3].split()]
    plane = np.asarray(lines)

    # Ensure normal is always facing up, this is in the rectified camera coordinate
    if plane[1] > 0:
        plane = -plane

    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm
    return plane

def _lidar_to_camera(pts, T_VELO_2_CAM, R_RECT_0):
    # pts [N,3]
    N = pts.shape[0]
    p = np.concatenate([pts, np.ones([N,1])], axis=1).T
    p = np.matmul(T_VELO_2_CAM, p)
    p = np.matmul(R_RECT_0, p)
    p = p[0:3]
    return p.T

def camera_to_lidar(pts, T_VELO_2_CAM, R_RECT_0):
    # pts [N,3]
    N = pts.shape[0]
    p = np.concatenate([pts, np.ones([N,1])], axis=1).T
    p = np.matmul(np.linalg.inv(R_RECT_0), p)
    p = np.matmul(np.linalg.inv(T_VELO_2_CAM), p)
    p = p[0:3]
    return p.T

def move_samples_to_road_plane(sampled_dict, road_plane, calib_dict):
    a, b, c, d = road_plane
    P, Tr_velo_to_cam, R_cam_to_rect = reorg_calib_dict(calib_dict)
    new_centers = []
    new_points = []
    for i in range(len(sampled_dict['points'])):
        new_center = sampled_dict['boxes_centers3d'][i].copy()
        new_point = sampled_dict['points'][i].copy()
        # lidar to camera
        cam_center = _lidar_to_camera(new_center[np.newaxis, :3], Tr_velo_to_cam, R_cam_to_rect)[0]
        cam_points = _lidar_to_camera(new_point[:,:3], Tr_velo_to_cam, R_cam_to_rect)
        # put it on the road plane
        cur_height = (-d - a * cam_center[0] - c * cam_center[2]) / b
        move_height = cam_center[1] - cur_height
        cam_center[1] -= move_height
        cam_points[:, 1] -= move_height
        # camera to lidar
        lidar_center = camera_to_lidar(cam_center[np.newaxis, ...], Tr_velo_to_cam, R_cam_to_rect)[0]
        lidar_center = np.concatenate([lidar_center, new_center[3:]])
        lidar_point = camera_to_lidar(cam_points, Tr_velo_to_cam, R_cam_to_rect)
        lidar_point = np.concatenate([lidar_point, new_point[:,3:4]], axis=1)
        # put it back
        new_centers.append(lidar_center)
        new_points.append(lidar_point)

    sampled_dict['boxes_centers3d'] = np.stack(new_centers, axis=0)
    sampled_dict['points'] = new_points
    return sampled_dict

if __name__ == '__main__':
    sampler = DataBaseSampler(root_path, database_info_path)
    sampler.print_class_name()
    # test1
    samples = sampler.sample_n_obj('Car', 11)
    # test2
    samples_points = sampler.load_sample_points(samples)
    # test3
    gt_boxes_corners3d = _get_gt()
    sampled = sampler.sample_all('Car', gt_boxes_corners3d)
    if sampled is not None:
        for i in range(len(sampled["names"])):
            print(sampled["points"][i].shape, sampled["names"][i])

        road_plane = get_road_plane(os.path.join(KITTI_PATH, 'planes', '%s.txt' % index))
        calib_dict = _get_calib_dict(index)
        sampled = move_samples_to_road_plane(sampled, road_plane, calib_dict)

        # (N, 7) -> (N, 8, 3)
        sampled_gt_boxes = sampled["boxes_centers3d"]
        print(sampled_gt_boxes.shape)
        sampled_boxes_corners3d = lidar_center_to_corner_box3d(sampled_gt_boxes)

        all_box3d = np.concatenate((gt_boxes_corners3d, sampled_boxes_corners3d), axis=0)
        view_pc(np.concatenate(sampled['points'], axis=0), all_box3d)

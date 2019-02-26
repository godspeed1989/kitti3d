import numpy as np
import math
import fire
from numbers import Number

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.occlusion = int(data[2])
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        # location (x,y,z) in camera coord.
        self.t = (data[11], data[12], data[13])
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.ry = data[14]


def read_label_obj(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

# camera center obj -> lidar corner (8,3)
def compute_lidar_box_3d(obj, R0, V2C):
    ''' Returns:
            corners_3d: (8,3) array in in lidar coord.
    '''
    P = 0
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    # 3d bounding box corners (in camera coord)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

    # (3,3) * (3,8)
    # project_rect_to_ref
    corners_3d = np.dot(np.linalg.inv(R0), corners_3d)
    corners_3d = np.transpose(corners_3d)

    # project_ref_to_velo
    corners_3d = cart2hom(corners_3d)
    C2V = inverse_rigid_trans(V2C)
    corners_3d = np.dot(corners_3d, np.transpose(C2V))

    return corners_3d

# (8, 3)[xyz] -> (7,)[x,y,z, l,w,h, rz]
def corner_to_center_box3d(corner):
    roi = np.array(corner)
    h = abs(np.sum(roi[:4, 2] - roi[4:, 2]) / 4)
    l = np.sum(
        np.sqrt(np.sum((roi[0, [0, 1]] - roi[3, [0, 1]])**2)) +
        np.sqrt(np.sum((roi[1, [0, 1]] - roi[2, [0, 1]])**2)) +
        np.sqrt(np.sum((roi[4, [0, 1]] - roi[7, [0, 1]])**2)) +
        np.sqrt(np.sum((roi[5, [0, 1]] - roi[6, [0, 1]])**2))
    ) / 4
    w = np.sum(
        np.sqrt(np.sum((roi[0, [0, 1]] - roi[1, [0, 1]])**2)) +
        np.sqrt(np.sum((roi[2, [0, 1]] - roi[3, [0, 1]])**2)) +
        np.sqrt(np.sum((roi[4, [0, 1]] - roi[5, [0, 1]])**2)) +
        np.sqrt(np.sum((roi[6, [0, 1]] - roi[7, [0, 1]])**2))
    ) / 4
    # x, y, z is at center
    x = np.sum(roi[:, 0], axis=0) / 8
    y = np.sum(roi[:, 1], axis=0) / 8
    z = np.sum(roi[:, 2], axis=0) / 8

    rz = np.sum(
        math.atan2(roi[2, 0] - roi[1, 0], roi[2, 1] - roi[1, 1]) +
        math.atan2(roi[6, 0] - roi[5, 0], roi[6, 1] - roi[5, 1]) +
        math.atan2(roi[3, 0] - roi[0, 0], roi[3, 1] - roi[0, 1]) +
        math.atan2(roi[7, 0] - roi[4, 0], roi[7, 1] - roi[4, 1]) +
        math.atan2(roi[0, 1] - roi[1, 1], roi[1, 0] - roi[0, 0]) +
        math.atan2(roi[4, 1] - roi[5, 1], roi[5, 0] - roi[4, 0]) +
        math.atan2(roi[3, 1] - roi[2, 1], roi[2, 0] - roi[3, 0]) +
        math.atan2(roi[7, 1] - roi[6, 1], roi[6, 0] - roi[7, 0])
    ) / 8

    return np.array([x, y, z, l, w, h, rz])

# for data augmentation
# rotate and translate points
def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])

    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)

    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)

    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)

    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)

    return points[:, 0:3]

def reorg_calib_dict(calib_dict):
    P = calib_dict['P2'].reshape(3, 4)
    P = np.concatenate((P, np.array([[0, 0, 0, 0]])), 0)
    #
    Tr_velo_to_cam = calib_dict['Tr_velo_to_cam'].reshape(3, 4)
    Tr_velo_to_cam = np.concatenate(
        [Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    #
    R_cam_to_rect = np.eye(4)
    R0_rect = calib_dict['R0_rect'].reshape(3, 3)
    R_cam_to_rect[:3, :3] = R0_rect
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

'''
This functions are for dump KITTI txt file
'''

# N*(x,y,z) -> N*(x,y,z)
def lidar_to_camera_point(points, T_VELO_2_CAM, R_RECT_0):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    # (4,N)
    points = np.hstack([points, np.ones((N, 1))]).T
    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)
# (x,y,z) -> (x,y,z)
def lidar_to_camera(x, y, z, T_VELO_2_CAM, R_RECT_0):
    # T_VELO_2_CAM 4x4
    # R_RECT_0 4x4
    p = np.array([x, y, z, 1])
    p = np.matmul(T_VELO_2_CAM, p)
    p = np.matmul(R_RECT_0, p)
    p = p[0:3]
    return tuple(p)

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    assert -np.pi/2 <= angle <= np.pi/2
    # if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
    #     angle = np.pi / 2
    return angle

def lidar_to_camera_box3d(boxes, T_VELO_2_CAM, R_RECT_0):
    # (N, 7) -> (N, 7) x,y,z, w,l,h,r
    ret = []
    for box in boxes:
        x, y, z, w, l, h, rz = box
        # warning: correct rz->ry should be : ry = -rz - np.pi / 2
        # there is some thing wrong in our corner_to_center_box3d()
        (x, y, z), w, l, h, ry = lidar_to_camera(x, y, z, T_VELO_2_CAM, R_RECT_0), \
                                    w, l, h, rz
        ry = angle_in_limit(ry)
        ret.append([x, y, z, w, l, h, ry])
    return np.array(ret).reshape(-1, 7)

# (N, 7) -> (N, 8, 3)
def lidar_center_to_corner_box3d(boxes_center):
    # coordinate(input): camera or lidar
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        w, l, h = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret

def lidar_box3d_to_camera_box(corners3d, cal_projection=False, P2=None, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 8, 3) -> (N, 4)        x1,y1,x2,y2
    # or
    # (N, 8, 3) -> (N, 8, 2)     8*(x, y)
    num = corners3d.shape[0]
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    for n in range(num):
        box3d = corners3d[n]
        box3d = lidar_to_camera_point(box3d, T_VELO_2_CAM, R_RECT_0)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = int(np.min(points[:, 0]))
        maxx = int(np.max(points[:, 0]))
        miny = int(np.min(points[:, 1]))
        maxy = int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d

def corners_2d_to_3d(corners2d, bottom_z, top_z):
    # (N, 4, 2) -> (N, 8, 3)
    # bottom_z, top_z: (1,) or (N,)
    N = corners2d.shape[0]
    corners3d = np.zeros((N, 8, 3), dtype=corners2d.dtype)
    corners3d[:,:4,:2] = corners2d
    corners3d[:,4:,:2] = corners2d
    corners3d[:,:4,2] = bottom_z[:, np.newaxis]
    corners3d[:,4:,2] = top_z[:, np.newaxis]
    return corners3d

def corners2d_to_3d(corners2d, bottom_z, top_z):
    # (N, 4, 2) -> (N, 7)
    # (N, 4, 2) -> (N, 8, 3)
    corners3d = corners_2d_to_3d(corners2d, bottom_z, top_z)
    N = corners3d.shape[0]
    center3d = []
    # (8, 3) -> (7,)
    for i in range(N):
        # (8, 3)[xyz] -> (7,)[x,y,z, l,w,h, rz]
        center = corner_to_center_box3d(corners3d[i])
        x, y = center[:2]
        z = center[2] - center[5]/2.0
        l, w, h, rz = center[3:]
        center3d.append([x,y,z, h,w,l, rz])
    return np.array(center3d), corners3d

def to_kitti_result_line(centers3d, corners3d, clses, scores, calib_dict):
    # Input:
    #   centers3d:  (N', 7) x y z l w h r
    #   scores:     float or (N')
    #   clses:      string or (N')      'Car' or 'Pedestrain' or 'Cyclist'
    # Output:
    #   label: (N') N' lines
    template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
    label = []
    N = len(centers3d)
    if isinstance(scores, Number): scores = [scores]*N
    if isinstance(clses, str): clses = [clses]*N
    P2, T_VELO_2_CAM, R_RECT_0 = reorg_calib_dict(calib_dict)
    #
    for center, corner, score, c in zip(centers3d, corners3d, scores, clses):
        box3d = lidar_to_camera_box3d(
            center[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
        box2d = lidar_box3d_to_camera_box(
            corner[np.newaxis, :].astype(np.float32), P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
        x, y, z, w, l, h, r = box3d
        #
        box3d = [w, l, h, x, y, z, r]
        label.append(template.format(c, 0, 0, 0, *box2d, *box3d, float(score)))

    return label

def test0():
    centers3d = np.array([[1,1,1,1,1,1,0], [2,2,2,1,1,1,0]])
    corners3d = np.random.rand(2, 8, 3) * 100
    clses = 'Car'
    scores = 0.5
    calib_dict = {}
    calib_dict['P2'] = np.random.rand(3, 4)
    calib_dict['Tr_velo_to_cam'] = np.random.rand(3, 4)
    calib_dict['R0_rect'] = np.random.rand(3, 3)
    labels = to_kitti_result_line(centers3d, corners3d, clses, scores, calib_dict)
    for line in labels:
        print(line, end='')

def test1():
    center = np.array([[1,3,5, 2,4,6, -1.3]], dtype=np.float32)
    center[:,6] = center[:,6]
    box3d = lidar_center_to_corner_box3d(center)
    print(box3d)
    center1 = corner_to_center_box3d(box3d[0])
    print(center1)  # [1,3,8, 4,2,6, 1.3]  ??

if __name__ == '__main__':
    fire.Fire()

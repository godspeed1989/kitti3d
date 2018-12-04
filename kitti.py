import numpy as np
import math

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


def compute_box_3d(obj, R0, V2C):
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


def to_kitti_result_line():
    return ''


# -- util function to load calib matrices
CAM = 2
def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [line.split()[1:] for line in lines][:-1]
    #
    P = np.array(lines[CAM]).reshape(3, 4)
    P = np.concatenate((P, np.array([[0, 0, 0, 0]])), 0)
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate(
        [Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

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
    rz = angle_in_limit(rz + np.pi / 2)

    return np.array([x, y, z, l, w, h, rz])

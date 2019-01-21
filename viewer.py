import os
import fire
import numpy as np
from OpenGL.GL import glLineWidth
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from kitti import read_label_obj, read_calib_file, compute_lidar_box_3d, corner_to_center_box3d

class plot3d(object):
    def __init__(self, title='null'):
        #
        self.glview = gl.GLViewWidget()
        coord = gl.GLAxisItem()
        glLineWidth(2)
        coord.setSize(1,1,1)
        self.glview.addItem(coord)
        self.glview.setMinimumSize(QtCore.QSize(600, 500))
        self.glview.pan(1, 0, 0)
        self.glview.setCameraPosition(azimuth=180)
        self.glview.setCameraPosition(elevation=0)
        self.glview.setCameraPosition(distance=5)
        self.items = []
        #
        self.view = QtGui.QWidget()
        self.view.window().setWindowTitle(title)
        hlayout = QtGui.QHBoxLayout()
        snap_btn = QtGui.QPushButton('&Snap')
        def take_snap():
            qimg = self.glview.readQImage()
            qimg.save('1.jpg')
        snap_btn.clicked.connect(take_snap)
        hlayout.addWidget(snap_btn)
        hlayout.addStretch()
        layout = QtGui.QVBoxLayout()
        #
        layout.addLayout(hlayout)
        layout.addWidget(self.glview)
        self.view.setLayout(layout)
    def add_item(self, item):
        self.glview.addItem(item)
        self.items.append(item)
    def clear(self):
        for it in self.items:
            self.glview.removeItem(it)
        self.items.clear()
    def add_points(self, points, colors):
        points_item = gl.GLScatterPlotItem(pos=points, size=0.5, color=colors)
        self.add_item(points_item)
    def add_line(self, p1, p2, color, width=3):
        lines = np.array([[p1[0], p1[1], p1[2]],
                          [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines',
                                       color=color, width=width, antialias=True)
        self.add_item(lines_item)
    def plot_bbox_mesh(self, gt_boxes3d, color=(0,1,0,1)):
        b = gt_boxes3d
        for k in range(0,4):
            i,j=k,(k+1)%4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)
            i,j=k+4,(k+1)%4 + 4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)
            i,j=k,k+4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)

def value_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = (pc_inte-minimum) / (maximum - minimum)
    r = (np.maximum((1 - ratio), 0))
    b = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b, np.ones_like(r)]).transpose()

SPLIT = 'training'
KITTI_PATH = '/mine/KITTI_DAT/' + SPLIT
KITTI_CALIB = '/mine/KITTI_DAT/calib/' + SPLIT

def plot_obj(viewer, objs, calib, color):
    for obj in objs:
        box3d_pts_3d = compute_lidar_box_3d(obj,
            calib['R0_rect'].reshape([3,3]),
            calib['Tr_velo_to_cam'].reshape([3,4]))
        viewer.plot_bbox_mesh(box3d_pts_3d, color)
        center = corner_to_center_box3d(box3d_pts_3d)
        print(center)
        min_xyz = center[:3] - center[3:6] / 2.0
        max_xyz = center[:3] + center[3:6] / 2.0
        viewer.add_line(min_xyz, max_xyz, (1,0,1,1))

def view(id='000018'):
    app = QtGui.QApplication([])

    f = os.path.join(KITTI_PATH, 'velodyne', id+'.bin')
    lidar = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    f = os.path.join(KITTI_PATH, 'label_2', id+'.txt')
    gt_obj = read_label_obj(f)
    f = os.path.join(KITTI_CALIB, id+'.txt')
    calib_dict = read_calib_file(f)

    glview = plot3d()
    # points cloud
    points = lidar[:,:3]
    intensity = lidar[:,3]
    points_color = value_to_rgb(intensity)
    glview.add_points(points, points_color)
    # gt bounding box
    plot_obj(glview, gt_obj, calib_dict, (0,1,0,1))
    # predict bounding box
    f = id + '.txt'
    if os.path.exists(f):
        pred_obj = read_label_obj(f)
        plot_obj(glview, pred_obj, calib_dict, (0,0,1,1))

    glview.view.show()
    return app.exec()

def view_points_cloud(pc=None):
    app = QtGui.QApplication([])
    glview = plot3d()
    if pc is None:
        pc = np.random.rand(1024, 3)
    pc_color = np.ones([pc.shape[0], 4])
    glview.add_points(pc, pc_color)
    glview.view.show()
    return app.exec()

def view_pc(pc=None, boxes3d=None):
    app = QtGui.QApplication([])
    glview = plot3d()
    if pc is None:
        points = np.random.rand(1024, 3)
        pc_color = np.ones([1024, 4])
    else:
        if pc.shape[1] == 3:
            points = pc[:,:3]
            pc_color = np.ones([pc.shape[0], 4])
        elif pc.shape[1] == 4:
            points = pc[:,:3]
            pc_color = value_to_rgb(pc[:,3])
    if boxes3d is not None:
        for box3d in boxes3d:
            glview.plot_bbox_mesh(box3d)
    glview.add_points(points, pc_color)
    glview.view.show()
    return app.exec()

if __name__ == "__main__":
    fire.Fire()
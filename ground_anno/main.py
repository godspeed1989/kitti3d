import os
import cv2
import numpy as np
from polyROISelector import orientedROISelector, plot_poly
from easydict import EasyDict


para = EasyDict()
para.L1 = -40.0
para.L2 = 40.0
para.W1 = 0.0
para.W2 = 70.4
para.H1 = -3
para.H2 = 1.0
para.grid_sizeLW = 0.1
para.input_shape = (800, 704)


KITTI_PATH = '/mine/KITTI_DAT'
VELO_SRC = KITTI_PATH + '/training/velodyne'

pdir = os.path.dirname(VELO_SRC)
ANNO_DST = os.path.join(pdir, 'grd_mask')

if not os.path.exists(ANNO_DST):
    os.makedirs(ANNO_DST)

# list all
path = os.path.join(KITTI_PATH, "train.txt")
with open(path, 'r') as f:
    lines = f.readlines() # get rid of \n symbol
    names = []
    for line in lines[:-1]:
        names.append(line[:-1])
    # Last line does not have a \n symbol
    names.append(lines[-1][:6])
print("There are {} images in txt file".format(len(names)))


def point_in_roi(point):
    if (point[0] - para.W1) < 0.01 or (para.W2 - point[0]) < 0.01:
        return False
    if (point[1] - para.L1) < 0.01 or (para.L2 - point[1]) < 0.01:
        return False
    if (point[2] - para.H1) < 0.01 or (para.H2 - point[2]) < 0.01:
        return False
    return True

def lidar_to_bev(velo):
    velo_processed = np.zeros((*para.input_shape, 3), dtype=np.float32)
    xs = ((velo[:, 1]-para.L1) / para.grid_sizeLW).astype(np.int32)
    ys = ((velo[:, 0]-para.W1) / para.grid_sizeLW).astype(np.int32)
    # import ipdb; ipdb.set_trace()
    for (x,y) in zip(xs, ys):
        if x < para.input_shape[0] and x >= 0 and \
           y < para.input_shape[1] and y >= 0:
            velo_processed[x, y, :] = 1
    return velo_processed

def get_lidar_img(idx):
    filename = os.path.join(VELO_SRC, names[idx]+'.bin')
    scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    bev = lidar_to_bev(scan)
    return bev

def line_to_poly(line):
    ret = []
    for s in line.split(' '):
        if s.isnumeric():
            ret.append(int(s))
    ret = np.array(ret, dtype=np.int32)
    return np.reshape(ret, [-1, 2])

def load_anno(idx):
    filename = os.path.join(ANNO_DST, names[idx]+'.txt')
    ret = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                pl = line_to_poly(l)
                ret.append(pl)
            f.close()
    return ret

def refresh(index, selector):
    image = get_lidar_img(index)
    clone = image.copy()
    old_anno = load_anno(index)
    for poly in old_anno:
        plot_poly(image, poly, (255, 0, 255))
    selector.resetCanvas(image)
    return image, clone, old_anno

def save_anno(selector, index, old_anno):
    filename = os.path.join(ANNO_DST, names[index]+'.txt')
    with open(filename, 'w') as f:
        for roi in selector.ROIs:
            for pt in roi['Polygon']:
                f.write('{} {} '.format(pt[0], pt[1]))
            f.write('\n')
        for poly in old_anno:
            for pt in poly:
                f.write('{} {} '.format(pt[0], pt[1]))
            f.write('\n')
        f.close()

ANNO_DST
index = 58
image = get_lidar_img(index)
clone = image.copy()
old_anno = load_anno(index)
for poly in old_anno:
    plot_poly(image, poly, (255, 0, 255))
windowName = "Anno"
cv2.imshow(windowName, image)

ROISelector = orientedROISelector(image, windowName=windowName)

num_frames = len(names)

while True:
    k = cv2.waitKey()
    if k == ord('r'):
        old_anno = []
        ROISelector.resetCanvas(clone.copy())
    elif k == 85: # up
        save_anno(ROISelector, index, old_anno)
        index = (index - 1 + num_frames) % num_frames
        print('==== {}: {} ===='.format(index, names[index]))
        image, clone, old_anno = refresh(index, ROISelector)
    elif k == 86: # down
        save_anno(ROISelector, index, old_anno)
        index = (index + 1) % num_frames
        print('==== {}: {}===='.format(index, names[index]))
        image, clone, old_anno = refresh(index, ROISelector)
    elif k == 27: # Esc
        break
    else:
        pass
cv2.destroyAllWindows()

import os
import numpy as np
import matplotlib.pyplot as plt

def removePoints(PointCloud, BoundaryCond):
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']
    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] > minX) & (PointCloud[:, 0] < maxX) & (PointCloud[:, 1] > minY) &
                    (PointCloud[:, 1] < maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]
    return PointCloud

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def makeBVFeature(PointCloud_origin, BoundaryCond, size_cell, dtype=np.float32):
    PointCloud = removePoints(PointCloud_origin, BoundaryCond)
    PointCloud[:, 0] = PointCloud[:, 0] - BoundaryCond['minX']
    PointCloud[:, 1] = PointCloud[:, 1] - BoundaryCond['minY']

    mark_ground = False
    if mark_ground:
        PointCloud_grd = extract_ground(PointCloud)
        PointCloud_grd[:, 0] = np.floor(PointCloud_grd[:, 0] / size_cell)
        PointCloud_grd[:, 1] = np.floor(PointCloud_grd[:, 1] / size_cell)
        PointCloud_grd[:, 0] = np.clip(PointCloud_grd[:, 0], 0, BoundaryCond['Height']-1)
        PointCloud_grd[:, 1] = np.clip(PointCloud_grd[:, 1], 0, BoundaryCond['Width']-1)

    # 将点云坐标转化为栅格坐标
    PointCloud[:, 0] = np.floor(PointCloud[:, 0] / size_cell)
    PointCloud[:, 1] = np.floor(PointCloud[:, 1] / size_cell)
    PointCloud[:, 0] = np.clip(PointCloud[:, 0], 0, BoundaryCond['Height']-1)
    PointCloud[:, 1] = np.clip(PointCloud[:, 1], 0, BoundaryCond['Width']-1)

    # np.lexsort((b,a)) 先对a排序，再对b排序
    # 按x轴(栅格）进行从小到大排列，当x值相同时，按y轴（栅格）从小到大排序，y也相同时，按z从大到小排序
    # 目的是将每个栅格的最大z排在最前面，下面unique时，便只会保留z最大值（排在第一位）的索引
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]
    # counts返回的是每个元素在原始数组出现的次数，这里是每个存在点的栅格中点的数量
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_frac = PointCloud[indices]

    # some important problem is image coordinate is (y,x), not (x,y)
    Width = BoundaryCond['Width']
    Height = BoundaryCond['Height']
    #
    heightMap = np.zeros((Width, Height), dtype=dtype)
    height_z = BoundaryCond['maxZ'] - BoundaryCond['minZ']
    heightMap[np.int_(PointCloud_frac[:, 1]), np.int_(PointCloud_frac[:, 0])] = \
        (PointCloud_frac[:, 2] - BoundaryCond['minZ']) / height_z
    heightMap = np.clip(heightMap, 0, 1.0)
    #
    densityMap = np.zeros((Width, Height), dtype=dtype)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))  # 即normalizedCounts最大为1
    densityMap[np.int_(PointCloud_frac[:, 1]), np.int_(PointCloud_frac[:, 0])] = normalizedCounts
    #
    intensityMap = np.zeros((Width, Height), dtype=dtype)
    indices = np.lexsort((-PointCloud[:, 3], PointCloud[:, 1], PointCloud[:, 0]))  # 按x由小到大，y由小到大，intensity由大到小
    PointCloud_intensity = PointCloud[indices]
    _, indices= np.unique(PointCloud_intensity[:, 0:2], axis=0, return_index=True)  # 只保留每个栅格中intensity最大的点
    PointCloud_max_intensity = PointCloud_intensity[indices]
    intensityMap[np.int_(PointCloud_max_intensity[:, 1]), np.int_(PointCloud_max_intensity[:, 0])] = \
        PointCloud_max_intensity[:, 3]
    intensityMap = np.clip(intensityMap, 0, 1.0)

    RGB_Map = np.stack([densityMap, heightMap, intensityMap], axis=2)

    if mark_ground:
        grdMap = np.ones((Width, Height), dtype=dtype)
        grdMap[np.int_(PointCloud_grd[:, 1]), np.int_(PointCloud_grd[:, 0])] = 0
        grdMap = np.expand_dims(grdMap, axis=-1)
        RGB_Map *= grdMap

    return RGB_Map

if __name__ == '__main__':
    size_ROI={}
    size_ROI['minX'] = 0; size_ROI['maxX'] = 70
    size_ROI['minY'] = -40; size_ROI['maxY'] = 40
    size_ROI['minZ'] = -2.5; size_ROI['maxZ'] = 1
    size_ROI['Height'] = 700
    size_ROI['Width'] = 800
    size_cell=0.1

    f = os.path.join('/mine/KITTI_DAT/training', 'velodyne', '000010'+'.bin')
    lidar = np.fromfile(f, dtype=np.float32).reshape(-1, 4)

    from datagen import extract_pc_in_fov
    pc, ind = extract_pc_in_fov(pc=lidar[:, :3], fov=50,
        X_MIN=0, X_MAX=70, Z_MIN=-2.5, Z_MAX=1)
    inte = lidar[ind, 3:]
    velo = np.concatenate((pc, inte), axis=1)

    RGB_Map = makeBVFeature(velo, size_ROI, size_cell)
    plt.imsave('bev.png', RGB_Map)
    plt.imshow(RGB_Map)
    plt.show()

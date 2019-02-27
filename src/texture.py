import imageio
import numpy as np


def idx_d2img(x, y, dd):
    '''
    Convert index from depth to rgb.

    Args:
        x (int): x index of depth image
        y (int): y index of depth image
        dd (float): dd

    Returns:
        rgb index
    '''
    img_x = (x * 526.37 + dd * (-4.5 * 1750.46) + 19276) / 585.051
    img_y = (y * 526.37 + 16662) / 585.051
    return img_x, img_y


def texture_mapping(grid, pose, res, rgb_path, d_path):
    '''
    Do texture mapping.

    Args:
        grid (np.ndaray): texture map
        pose (np.ndarray): pose after particle filtering (Must be a .copy() of origin)
        res (float): resolution
        rgb_path (str): file path of rgb image
        d_path (str): file path of depth image

    Returns:
        Texture map (np.ndarray)
    '''
    # convert to meters instead of map resolution
    pose[0] = pose[0] * res
    pose[1] = pose[1] * res

    # angle
    roll = 0
    pitch = 0.36
    yaw = 0.021

    d = imageio.imread(d_path)
    rgb_img = imageio.imread(rgb_path)

    # compute dd and depth
    dd = -0.00304 * d+3.31
    depth = 1.03/dd

    # set threshold
    valid = np.where(np.logical_and(depth < 5, depth > 0.05))

    # coordinates of depth image
    depth_i = valid[0]
    depth_j = valid[1]

    # Intrinsic matrix
    intrinsic = np.array([[585.05708211, 0, 242.94140713],
                          [0, 585.05108211, 315.83800193],
                          [0, 0, 1]])

    # Rotation from optical frame to camera frame
    cRo = np.zeros((3, 3))
    cRo[0, 1] = -1
    cRo[1, 2] = -1
    cRo[2, 0] = 1

    # Rotation: camera frame to body frame
    bRc_x = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])
    bRc_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
    bRc_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    bRc = (bRc_z.dot(bRc_y)).dot(bRc_x)

    # Position: camera frame to body frame
    bPc = np.array([0.18, 0.005, 0.36]).T

    # Transform: camera frame to body frame
    bTc = np.zeros((4, 4))
    bTc[0:3, 0:3] = bRc
    bTc[0:3, 3] = bPc
    bTc[3, 3] = 1

    # Rotation: body frame to world frame
    theta = pose[2]
    wRb = np.zeros((3, 3))
    wRb[0, 0] = np.cos(theta)
    wRb[0, 1] = -np.sin(theta)
    wRb[1, 0] = np.sin(theta)
    wRb[1, 1] = np.cos(theta)
    wRb[2, 2] = 1

    # Transform: body frame to world frame
    wTb = np.zeros((4, 4))
    wTb[0:3, 0:3] = wRb
    wTb[0:3, 3] = pose
    wTb[3, 3] = 1

    wTc = bTc.dot(wTb)
    cTw = np.linalg.inv(wTc)

    cRw = wTc[0:3, 0:3]
    cPw = wTc[0:3, 3]

    # extrinsic matrix (4 x 4)
    extrinsic = np.zeros((4, 4))
    extrinsic[0:3, 0:3] = cRo.dot(cRw.T)
    extrinsic[0:3, 3] = -cRo.dot(cRw.T).dot(cPw)
    extrinsic[3, 3] = 1

    for idx in range(depth_i.size):
        # Pixels (Image frame)
        rgb_i, rgb_j = idx_d2img(
            depth_i[idx], depth_j[idx], dd[depth_i[idx], depth_j[idx]])
        pixel = np.array([rgb_i, rgb_j, 1]).T
        Z0 = depth[depth_i[idx], depth_j[idx]]

        # canonical projection
        cano = np.zeros((3, 3))
        cano[0, 0] = 1/Z0
        cano[1, 1] = 1/Z0
        cano[2, 2] = 1/Z0

        # optical coordinates
        op_coord = np.linalg.inv(cano).dot(
            np.linalg.inv(intrinsic)).dot(pixel)
        op_coord = np.concatenate((op_coord, np.array([1])))

        # world coordinates
        world_coord = np.linalg.inv(extrinsic).dot(op_coord)
        grid[int(world_coord[0]/res)+grid.shape[0]//2, int(world_coord[1] /
                                                           res)+grid.shape[1]//2, :] = rgb_img[int(rgb_i), int(rgb_j), :]

    return grid

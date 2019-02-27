import numpy as np
from numpy.random import rand
from map_utils import bresenham2D


def pol2cart(r, theta):
    '''
    Convert coordinates from polar to Cartesian.
    Theta ranges from -135 to 135.

    Args:
        r: radius (in meter), each sample has 1081 points

    Returns:
        x: x-coord of lidar scan
        y: y-coord of lidar scan
    '''
    # starting angle in radian
    start = -135/180*np.pi
    theta = np.arange(start, -start, -2*start/1081) + theta
    x, y = r * np.cos(theta), r * np.sin(theta)
    return x, y


def mapping(grid, pose, scan, res, empty_odds, ocpy_odds):
    '''
    Update the grid map with log-odds.

    Args:
        grid (80, 80): grid map
        scan (1081,): lidar ranges
        pose (3,):  pose (x,y,theta)
        res (float or int): resolution
        empty_odds (float): empty odds
        ocpy_odds (float): occupied odds

    Returns:
        grid map (80, 80)
    '''
    # pixel value
    value = 127

    # convert from polar to Cartesian
    x, y = pol2cart(scan, pose[2])  # 1-d array

    # discretization (1-d array)
    xi = (x / res).astype(int)
    yi = (y / res).astype(int)

    wall_set = np.zeros_like(grid)
    empty_set = np.zeros_like(grid)
    for i in range(len(xi)):
        a, b = xi.item(i), yi.item(i)
        x1 = a + int(pose[0]) + grid.shape[0]//2
        y1 = b + int(pose[1]) + grid.shape[1]//2
        if 0 <= x1 < grid.shape[0] and 0 <= y1 < grid.shape[1]:
            wall_set[x1, y1] = 1

        line = np.array(bresenham2D(0, 0, a, b)).astype(int)
        for j in range(len(line[0]) - 1):
            x2 = line[0][j] + int(pose[0]) + grid.shape[0]//2
            y2 = line[1][j] + int(pose[1]) + grid.shape[1]//2
            if 0 <= x2 < grid.shape[0] and 0 <= y2 < grid.shape[1]:
                empty_set[x2, y2] = 1

    grid[wall_set == 1] += ocpy_odds
    grid[empty_set == 1] += empty_odds

    grid[grid >= value] = value
    grid[grid < -value] = -value

    return grid


def softmax(z):
    '''
    Compute softmax of `z`.

    Args:
        z (1-d array): data

    Returns:
        Softmax of `z`.
    '''
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def stratified_resample(w):
    '''
    Stratified Sample.

    Args:
        w: weight

    Returns:
        Index
    '''
    N = len(w)
    rng = (rand(N) + range(N)) / N

    cum_w = np.cumsum(w)
    result = []
    i, j = 0, 0
    while i < N:
        if rng[i] < cum_w[j]:
            result.append(j)
            i += 1
        else:
            j += 1
    return np.array(result)

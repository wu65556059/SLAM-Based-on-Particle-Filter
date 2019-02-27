import os
import numpy as np
from numpy.random import randn
from map_utils import (mapCorrelation, bresenham2D)
from core import (mapping, softmax, pol2cart, stratified_resample)
from texture import texture_mapping

class ParticleFilter():
    def __init__(self, dataset, grid_size, pose, scan, noise, step, N, ocpy_odds, empty_odds):
        self.dataset = dataset
        # map
        self.res = 0.1  # resolution
        self.grid_size = grid_size
        self.map_center = grid_size // 2
        self.grid = np.zeros((grid_size, grid_size))
        self.texture_map = np.zeros((grid_size, grid_size, 3))
        # location
        self.step = step
        self.walk_loc = np.zeros((int(scan.shape[1]/step)+1, 2))
        self.slam_loc = np.zeros((int(scan.shape[1]/step)+1, 2))
        # parameters
        self.noise = noise
        self.N = N  # number of particles
        self.particles = np.zeros((N, 3))
        self.W = np.ones(N) / N  # initialize same prior
        self.empty_odds = empty_odds
        self.ocpy_odds = ocpy_odds
        self.lidar_angles = np.array([i * 0.0044 for i in range(-540, 541, 1)])

        # 1-d array
        self.cor_mle = np.zeros(N)

        self.grid = mapping(
            self.grid, pose[0], scan[:, 0], self.res, self.empty_odds, self.ocpy_odds)
        self.prev_pose = pose[0]

       
    def predict(self, pose, scan):
        '''
        Prediction of particle filter.

        Args:
            pose: encoder pose (x, y, theta)
            scan: lidar ranges

        Returns:
            Updated `cor_mle`.
        '''
        # particles
        pose_diff = pose - self.prev_pose
        self.prev_pose = pose
        self.particles += pose_diff + randn(self.N, 3) * self.noise
        self.particles[:, 2] %= 2 * np.pi
        self.cor_mle = np.zeros(self.N)

        # index of x and y coordinate
        x_im, y_im = np.arange(
            self.grid.shape[0]), np.arange(self.grid.shape[1])

        # 5x5 window
        xs, ys = np.arange(-2*self.res, 3*self.res,
                           self.res), np.arange(-2*self.res, 3*self.res, self.res)

        # temp grid for computing map correlation
        temp = np.zeros_like(self.grid)
        temp[self.grid > 0] = 1
        temp[self.grid < 0] = -1

        # iterate N particles
        for j in range(self.N):
            # pose in world frame
            theta = (self.lidar_angles + self.particles[j][2]).reshape((-1, 1))
            x = scan * np.cos(theta) / self.res + self.grid.shape[0]//2
            y = scan * np.sin(theta) / self.res + self.grid.shape[1]//2
            # map correlation
            mpcor = mapCorrelation(temp, x_im, y_im, np.vstack(
                (x, y)), self.particles[j][0]+xs, self.particles[j][1]+ys)
            self.cor_mle[j] = np.max(mpcor)

    def update(self, pose, scan, i):
        '''
        Update of particle filter.

        Args:
            pose: encoder pose
            scan: lidar ranges
            i: `ith` iteration

        Returns:
            Updated `W`, `grid`, `slam_map`, and `walk_map`.
        '''
        # update weight
        self.cor_mle = self.W * self.cor_mle
        self.W = softmax(self.cor_mle)

        best_idx = np.argmax(self.W)  # np.where returns a tuple
        pf_pose = self.particles[best_idx]  # .copy()
        self.grid = mapping(self.grid, pf_pose, scan,
                            self.res, self.empty_odds, self.ocpy_odds)

        # update texture map
        rgb_name = os.getcwd() + \
            f'/data/dataRGBD/RGB{self.dataset}/rgb{self.dataset}_{i//self.step+1}.png'
        depth_name = os.getcwd() + \
            f'/data/dataRGBD/Disparity{self.dataset}/disparity{self.dataset}_{i//self.step+1}.png'
        self.texture_map = texture_mapping(
            self.texture_map, pf_pose.copy(), self.res, rgb_name, depth_name)

        n_eff = 1 / (self.W**2).sum()
        if n_eff < 0.85 * self.N:
            idx = stratified_resample(self.W)
            self.particles[:] = self.particles[idx]
            self.W.fill(1.0 / self.N)

        # update point coordinates
        self.slam_loc[i//self.step] = np.array([int(pf_pose[0]) + self.slam_map.shape[0]//2,
                                                int(pf_pose[1]) + self.slam_map.shape[1]//2]).T
        self.walk_loc[i//self.step] = np.array([int(pose[0]) + self.walk_map.shape[0]//2,
                                                int(pose[1]) + self.walk_map.shape[1]//2]).T


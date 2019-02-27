import numpy as np
import matplotlib.pyplot as plt
from particle_filter import ParticleFilter
from data_loader import DataLoader


def run_SLAM(ds, iter_step, N, ocpy_odds, empty_odds):
    '''
    Implement SLAM.

    Args:
        ds (int): dataset
        iter_step (int): step for the time stamps
        N (list): noise
        ocpy_odds (float): occupancy log odds
        empty_odds (float): empty log odds
    '''

    # load
    data = DataLoader(ds)
    noise = np.array([0.25, 0.25, 0.1*np.pi / 180])
    res = 0.1
    grid_size = int(80 / res)
    lidar_rng = data.lidar_ranges

    # encoder trajectory in meters
    encoder_x, encoder_y, theta = data.encoder_x, data.encoder_y, data.encoder_theta

    # number of data
    num = max(encoder_x.shape)

    # current pose in pixels (/res)
    encoder_pose = np.hstack((encoder_x/res, encoder_y/res, theta))

    # initialize particle filter
    pf = ParticleFilter(ds, grid_size, encoder_pose, lidar_rng,
                        noise, iter_step, N, ocpy_odds, empty_odds)

    # iterate trajectory points
    for i in range(0, num, iter_step):
        print(
            f'\ndataset:{ds}|N:{N}|opcy_odds:{ocpy_odds:.2f}|Progress:{i/num*100:.2f}%')
        pf.predict(encoder_pose[i], lidar_rng[:, i])
        pf.update(encoder_pose[i], lidar_rng[:, i], i)

        if i % 460 == 0:
            # save the SLAM progress
            fig = pf.grid.copy()
            fig[pf.grid > 0] = 2
            fig[np.logical_and(0 <= pf.grid, pf.grid <= 0)] = 0
            fig[pf.grid < 0] = 0.5
            plt.figure()
            plt.axis('off')
            plt.imshow(fig, cmap='gray')
            plt.savefig(
                f'img/result/ds{ds}_lidar_step{iter_step}_n_{noise[0]}_ang_{noise[2]:.5f}_log_{ocpy_odds:.2f}_N_{N}_i_{i}.png')

    # plot
    fig = pf.grid.copy()
    fig[pf.grid > 0] = 2
    fig[np.logical_and(0 <= pf.grid, pf.grid <= 0)] = 0
    fig[pf.grid < 0] = 0.5

    # lidar only
    plt.figure()
    plt.axis('off')
    plt.imshow(fig, cmap='gray')
    plt.savefig(
        f'img/result/ds{ds}_lidar_step{iter_step}_n_{noise[0]}_ang_{noise[2]:.5f}_log_{ocpy_odds:.2f}_N_{N}.png')

    # lidar with trajectory
    plt.figure()
    plt.axis('off')
    plt.plot(pf.slam_loc[:, 1], pf.slam_loc[:, 0], 'r-')
    plt.plot(pf.walk_loc[:, 1], pf.walk_loc[:, 0], 'b-')
    plt.imshow(fig, cmap='gray')

    # texture map
    plt.figure()
    plt.axis('off')
    plt.imshow(pf.texture_map.astype(np.uint8))
    plt.savefig(f'img/result/ds{ds}_texture_step{iter_step}_n_{noise[0]}.png')

    plt.show()
    plt.close()


if __name__ == '__main__':
    dataset = [21, 23, 20]
    ocpy_odds = [np.log(0.9/0.1), np.log(0.8/0.2), np.log(0.7/0.3)]
    empty_odds = [np.log(0.1/0.9), np.log(0.2/0.8), np.log(0.3/0.7)]
    iter_step = 20
    N = [200, 100]
    for ds in dataset:
        for odd in range(len(ocpy_odds)):
            for n in range(len(N)):
                run_SLAM(ds, iter_step, N[n], ocpy_odds[odd], empty_odds[odd])

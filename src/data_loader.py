import numpy as np
import os


class DataLoader():
    def __init__(self, dataset):
        assert(isinstance(dataset, int))
        file_path = os.getcwd() + '/data/'

        with np.load(file_path+"Encoders%d.npz" % dataset) as data:
            self.encoder_counts = data["counts"]  # 4 x n encoder counts
            self.encoder_stamps = data["time_stamps"]  # encoder time stamps

        with np.load(file_path+"Hokuyo%d.npz" % dataset) as data:
            # start angle of the scan [rad]
            self.lidar_angle_min = data["angle_min"]
            # end angle of the scan [rad]
            self.lidar_angle_max = data["angle_max"]
            # angular distance between measurements [rad]
            self.lidar_angle_increment = data["angle_increment"]
            self.lidar_range_min = data["range_min"]  # minimum range value [m]
            self.lidar_range_max = data["range_max"]  # maximum range value [m]
            # range data [m] (Note: values < range_min or > range_max should be discarded)
            self.lidar_ranges = data["ranges"]
            # acquisition times of the lidar scans
            self.lidar_stamps = data["time_stamps"]

        with np.load(file_path+"Imu%d.npz" % dataset) as data:
            # angular velocity in rad/sec
            self.imu_angular_velocity = data["angular_velocity"]
            # Accelerations in gs (gravity acceleration scaling)
            self.imu_linear_acceleration = data["linear_acceleration"]
            # acquisition times of the imu measurements
            self.imu_stamps = data["time_stamps"]

        if dataset != 23:  # not load if test set
            with np.load(file_path+"Kinect%d.npz" % dataset) as data:
                # acquisition times of the disparity images
                self.disp_stamps = data["disparity_time_stamps"]
                # acquisition times of the rgb images
                self.rgb_stamps = data["rgb_time_stamps"]

        # align encoder data w.r.t. lidar time stamps
        e_idx = self.__time_align(self.encoder_stamps)
        self.encoder_stamps = self.encoder_stamps[e_idx]
        self.encoder_counts = self.encoder_counts[:, e_idx]

        # align imu data w.r.t. lidar time stamps
        imu_idx = self.__time_align(self.imu_stamps)
        self.imu_stamps = self.imu_stamps[imu_idx]
        self.imu_angular_velocity = self.imu_angular_velocity[2, imu_idx]

        self.encoder_x, self.encoder_y, self.encoder_theta = self.__encoder_trajectory()

        # align rgbd data w.r.t. lidar time stamps
        if dataset != 23:  # not load if test set
            rgbd_idx = self.__time_align(self.rgb_stamps)
            d_idx = self.__time_align(self.disp_stamps)
            self.rgb_stamps = self.rgb_stamps[rgbd_idx]
            self.disp_stamps = self.disp_stamps[d_idx]

    def __time_align(self, time):
        '''
        Align time stamps with respect to encoder. 

        Args:
            time (np.ndarray): epoch time

        Returns:
            Aligned index of `time` (1d-array)
        '''
        num = len(self.lidar_stamps)
        idx = np.zeros(num, dtype=int)
        for i, v in enumerate(self.lidar_stamps):
            idx[i] = np.argmin(abs(v - time))
        return idx

    def __encoder_trajectory(self):
        '''
        Compute x and y coordinates of encoder trajectory in world frame.

        Args:
            counts (np.ndarray): encoder counts
                  | 0  |  1  | ...
                --|----|-----|----
                 0| FR | ... | ...
                --|----|-----|----
                 1| FL | ... | ...
                --|----|-----|----        
                 2| RR | ... | ...
                --|----|-----|----
                 3| RL | ... | ...

            w (np.ndarray): angular velocity from IMU

            e_time (np.ndarray): time stamps from encoder

        Returns:
            x (np.ndarray (N, 1))
            y (np.ndarray (N, 1))
            theta (np.ndarray (N, 1))
        '''
        counts = self.encoder_counts
        w = self.imu_angular_velocity
        e_time = self.encoder_stamps

        d_t = np.diff(e_time)
        # t = np.add.accumulate(d_t, axis=1)

        counts = counts[:, 1:]
        w = w[1:]

        # distance of right center for each move in meters
        d_r = ((counts[0]+counts[2]) / 2 * 0.0022).reshape(1, -1)
        d_r = np.hstack((np.zeros((1, 1)), d_r))
        # distance of left center for each move in meters
        d_l = ((counts[1]+counts[3]) / 2 * 0.0022).reshape(1, -1)
        d_l = np.hstack((np.zeros((1, 1)), d_l))
        # distance of body center for each move in meters
        d_c = (d_r + d_l) / 2

        # Compute theta from IMU
        d_theta = w * d_t
        theta = np.add.accumulate(d_theta)
        theta = np.hstack((0, theta))

        # Compute v from encoder
        # vr = d_r / d_t
        # vl = d_l / d_t
        # v = 0.5 * (vr + vl)

        # compute x and y
        x = np.add.accumulate(d_c * np.cos(theta), axis=1)
        y = np.add.accumulate(d_c * np.sin(theta), axis=1)

        return x.T, y.T, theta.reshape((-1, 1))

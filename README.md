# Simultaneous Localization and Mapping (SLAM) based on Particle Filter

This project implements Simultaneous Localization and Mapping (SLAM) using odometry, inertial, 2-D laser range, and RGBD measurements from a differential-drive robot, shown below, based on particle filter.

![](img/robot.png)

## Requirements

The following are packages that I used:

* `numpy` >= 1.16.1  

* `imageio` >= 2.5.0  

* `matplotlib` >= 3.0.2  

```
pip3 install -r requirements.txt
```

## Testing Environment

* sysname: `Linux`  

* release: `4.15.0-45-generic`  

* machine: `x86_64`  

* python: `3.6.7` and `3.7.1`

## Result

### Training Data 1

| Trajectory | SLAM Progress |  
|:-:|:-:|  
| ![](img/ds20/ds_20_lidar_trj.png) | ![](img/ds20/ds_20.gif) | 

### Training Data 2

| Trajectory | SLAM Progress |  
|:-:|:-:|  
| ![](img/ds21/ds_21_lidar_trj.png) | ![](img/ds21/ds_21.gif) | 

### Testing Data

| Trajectory | SLAM Progress |  
|:-:|:-:|  
| ![](img/ds23/ds_23_lidar_trj.png) | ![](img/ds23/ds_23.gif) | 


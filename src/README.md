# Description

## main.py

Implement SLAM based on particle filter.

## core.py

* `pol2cart()`: Convert coordinates from polar to Cartesian.

* `mapping()`: Update the grid map with log-odds.

* `softmax()`: Compute softmax function.

* `stratified_resample()`: Implement stratified sample.

## data_loader.py

`DataLoader()`: Load data and preprocess data, such as time stamps alignment and compute initial trajectory.

## particle_filter.py

`ParticleFilter()`: Implement predict and update steps of particle filter.

## texture.py

* `idx_d2img()`: Convert index from depth to rgb.

* `texture_mapping()`: Implement texture mapping.
# Description

## core.py

* `pol2cart()`: Convert coordinates from polar to Cartesian.

* `mapping()`: Update the grid map with log-odds.

* `softmax()`: Compute softmax function.

* `stratified_resample()`: Implement stratified sample.

## data_loader.py

Load data set to a class `DataLoader()`, and preprocess data, such as time stamps alignment and compute initial trajectory.

## particle_filter.py

Implement predict and update steps of particle filter in a class `ParticleFilter()`.

# texture.py

* `idx_d2img()`: Convert index from depth to rgb.

* `texture_mapping()`: Implement texture mapping.
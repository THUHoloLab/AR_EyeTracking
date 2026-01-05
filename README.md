# AR_EyeTracking

This repository implements an iterative image reconstruction algorithm for spatially variant systems (e.g., AR waveguides). 
It uses a **Low-Rank approximation** of the Point Spread Function (PSF) to accelerate the forward and backward operators.

## Features

- **Forward Model**: Spatially variant convolution approximated by Singular Value Decomposition (SVD).
- **Optimization**: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with Nesterov momentum.
- **Regularization**: Total Variation (TV) using MATLAB's `dlarray` automatic differentiation.
- **Acceleration**: Supports GPU acceleration.

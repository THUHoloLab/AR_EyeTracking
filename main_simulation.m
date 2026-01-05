%% ========================================================================
%  Low-Rank PSF Reconstruction Demo (Simulation Mode)
%  ========================================================================
%  Description:
%   1. Loads Ground Truth (GT) from LPW dataset.
%   2. Simulates degradation using a Spatially Variant PSF (Low-Rank Approx).
%   3. Reconstructs the image using FISTA + Total Variation (TV).
%
%  Dependencies:
%   - Deep Learning Toolbox (for dlarray/dlgradient)
%   - Parallel Computing Toolbox (optional, for GPU)
% ========================================================================

clear; clc; close all;
addpath('utils');

% --- Configuration ---
settings.gpu_id = 1;            % Set to 0 for CPU, 1 for GPU
settings.lambda = 0.01;         % Regularization weight for TV
settings.num_iters = 200;       % Number of FISTA iterations
settings.rank_num = 40;         % Number of Low-Rank components to use
settings.img_size = [486, 648]; % Target height/width

% --- 1. Environment Setup ---
if settings.gpu_id > 0 && canUseGPU()
    g = gpuDevice(settings.gpu_id);
    reset(g);
    fprintf('Device: GPU (%s)\n', g.Name);
    use_gpu = true;
else
    fprintf('Device: CPU\n');
    use_gpu = false;
end

%% --- 2. Load and Prepare PSF Model ---
fprintf('Loading PSF Model...\n');
try
    model_data = load(fullfile('data', 'models', 'CroppedPSF_GT_z=5.5cm.mat'));
catch
    error('Model file not found. Please ensure "CroppedPSF_GT_z=5.5cm.mat" is in "data/models/".');
end

% Prepare System Matrix (Low-Rank Decomposition)
[all_OTFs, HR_weights] = physics_model.prepare_lowrank_model(...
    model_data, settings.img_size, settings.rank_num, use_gpu);

% Calculate Lipschitz constant (Step Size)
fprintf('Calculating Step Size...\n');
L = math_utils.power_iteration(all_OTFs, HR_weights, settings.img_size);
step_size = 0.9 / L;
fprintf('Lipschitz L: %.4e | Step Size: %.4e\n', L, step_size);

%% --- 3. Load Data & Simulate Degradation ---
% Define image path
img_path = fullfile('data', 'test_images', 'sample_lpw.png'); 

% Strict validation: Check if file exists
if ~exist(img_path, 'file')
    error('Input Error: Image file not found at "%s". Script terminated.', img_path);
end

% Attempt to read and process the image
fprintf('Loading Ground Truth image: %s\n', img_path);
try
    gt_raw = imread(img_path);
catch ME
    error('Read Error: Failed to read image "%s". \nReason: %s', img_path, ME.message);
end

% Preprocessing (Grayscale -> Resize -> Normalize)
if size(gt_raw, 3) == 3
    gt_img = rgb2gray(gt_raw);
else
    gt_img = gt_raw;
end

gt_img = im2double(gt_img);
gt_img = imresize(gt_img, settings.img_size);
gt_img = gt_img / max(gt_img(:)); % Normalize to [0, 1]

% Transfer to GPU if enabled
if use_gpu
    gt_gpu = gpuArray(gt_img); 
else
    gt_gpu = gt_img; 
end

% Forward Simulation (Blur)
fprintf('Simulating optical degradation...\n');
sim_blur = physics_model.forward(gt_gpu, all_OTFs, HR_weights);

% Add Noise (Gaussian)
sim_degraded = imnoise(sim_blur, 'gaussian', 0, 1e-4);

%% --- 4. Reconstruction (FISTA + TV) ---
fprintf('Starting Reconstruction...\n');

% Initialization
x_k = zeros(settings.img_size, 'like', sim_degraded); % Estimate
y_k = x_k; % Momentum variable
t_k = 1;
loss_history = zeros(settings.num_iters, 1);

tic;
for k = 1:settings.num_iters
    % 1. Data Fidelity Gradient (Adjoint Operator)
    m_sim = physics_model.forward(y_k, all_OTFs, HR_weights);
    residual = m_sim - sim_degraded;
    grad_data = physics_model.adjoint(residual, all_OTFs, HR_weights);
    
    % 2. Regularization Gradient (AutoDiff TV)
    % Convert to dlarray for auto-differentiation
    y_k_dl = dlarray(y_k); 
    [tv_val, grad_reg_dl] = dlfeval(@tv_regularizer.compute_gradient, y_k_dl);
    grad_reg = settings.lambda * extractdata(grad_reg_dl);
    
    % 3. Gradient Descent Step
    grad_total = grad_data + grad_reg;
    x_next = y_k - step_size * grad_total;
    
    % 4. Proximal Operator (Non-negativity constraint)
    x_next = max(x_next, 0);
    
    % 5. Nesterov Momentum Update (FISTA)
    t_next = (1 + sqrt(1 + 4 * t_k^2)) / 2;
    beta = (t_k - 1) / t_next;
    y_k = x_next + beta * (x_next - x_k);
    
    % Update variables
    x_k = x_next;
    t_k = t_next;
    
    % Logging
    loss_data = 0.5 * norm(residual(:))^2;
    loss_reg = settings.lambda * extractdata(tv_val);
    loss_history(k) = gather(loss_data + loss_reg);
    
    if mod(k, 20) == 0
        fprintf('Iter %3d | Loss: %.4e \n', k, loss_history(k));
    end
end
recon_time = toc;
img_recon = gather(x_k);
fprintf('Reconstruction finished in %.2f seconds.\n', recon_time);

%% --- 5. Visualization ---
figure('Name', 'Reconstruction Results', 'Color', 'w', 'Position', [100, 100, 1200, 400]);

% Ground Truth
subplot(1, 3, 1); imagesc(gt_img); axis image off; colormap gray;
title('Ground Truth (LPW)');

% Degraded
subplot(1, 3, 2); imagesc(gather(sim_degraded)); axis image off; colormap gray;
title('Simulated Degraded');

% Reconstructed
subplot(1, 3, 3); imagesc(img_recon); axis image off; colormap gray;
title(['Reconstruction (TV \lambda=', num2str(settings.lambda), ')']);

% Save result
save_path = 'result_recon.png';
imwrite(img_recon, save_path);
fprintf('Result saved to %s\n', save_path);
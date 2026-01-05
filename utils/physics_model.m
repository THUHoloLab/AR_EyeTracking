classdef physics_model
    methods(Static)
        
        function [all_OTFs, HR_weights] = prepare_lowrank_model(model_data, img_size, LR_num, use_gpu)
            % Prepares the Optical Transfer Functions (OTF) and Spatial Weights
            
            % Unpack Model
            GT_pixelspace = reshape(model_data.GT_pixelspace, [21, 21, 2]);
            A_full = reshape(model_data.croppedPSFArray, [441, 22500]);
            
            % Normalize
            psf_energies = sum(abs(A_full), 2);
            A_full = A_full / max(psf_energies);
            A_full = A_full'; % [Pixels x Positions]
            
            % SVD Decomposition
            [U, S, V] = svd(A_full, 'econ');
            VT = V';
            
            % Extract Low-Rank Components
            U_LR = U(:, 1:LR_num);
            S_LR = S(1:LR_num, 1:LR_num);
            VT_LR = VT(1:LR_num, :);
            
            % Basis PSFs -> OTFs
            PSF_LR = reshape(U_LR * S_LR, [150, 150, LR_num]);
            all_OTFs_cpu = zeros(img_size(1), img_size(2), LR_num);
            for i = 1:LR_num
                % psf2otf handles padding and centering automatically if implemented, 
                % otherwise use fft2 with padding.
                all_OTFs_cpu(:,:,i) = psf2otf(squeeze(PSF_LR(:,:,i)), img_size);
            end
            
            % Spatial Weights Interpolation
            weight_LR = reshape(VT_LR, [LR_num, 21, 21]);
            xi = 1:img_size(1); yi = 1:img_size(2);
            [Xi, Yi] = ndgrid(xi, yi);
            
            % Grid definition based on your calibration
            zeros_bound = 1;
            x_lr = [0+zeros_bound; squeeze(GT_pixelspace(:, 1, 2)); img_size(1)];
            y_lr = [0+zeros_bound, squeeze(GT_pixelspace(1, :, 1)), img_size(2)];
            
            HR_weights_cpu = zeros(LR_num, img_size(1), img_size(2));
            
            % Pad weights for interpolation
            weight_LR_perm = zeros(LR_num, 23, 23);
            weight_LR_perm(:, 2:22, 2:22) = weight_LR;
            
            for i = 1:LR_num
                HR_weights_cpu(i,:,:) = interpn(x_lr, y_lr, squeeze(weight_LR_perm(i,:,:)), Xi, Yi, 'linear');
            end
            
            % GPU Transfer
            if use_gpu
                all_OTFs = gpuArray(all_OTFs_cpu);
                HR_weights = gpuArray(permute(HR_weights_cpu, [2, 3, 1])); % [H, W, Rank]
            else
                all_OTFs = all_OTFs_cpu;
                HR_weights = permute(HR_weights_cpu, [2, 3, 1]);
            end
        end

        function m = forward(o, all_OTFs, weights)
            % Forward Model: A * x
            % Spectral convolution followed by spatial weighting sum
            fft_o = fft2(o);
            conv_freq = fft_o .* all_OTFs;
            conv_spatial = real(ifft2(conv_freq));
            m = sum(weights .* conv_spatial, 3);
        end
        
        function o_adj = adjoint(m, all_OTFs, weights)
            % Adjoint Model: A^T * x
            % Spatial weighting followed by correlation (conj convolution)
            weighted_m = m .* weights;
            weighted_m_fft = fft2(weighted_m);
            o_adj_fft_sum = sum(weighted_m_fft .* conj(all_OTFs), 3);
            o_adj = real(ifft2(o_adj_fft_sum));
        end
    end
end
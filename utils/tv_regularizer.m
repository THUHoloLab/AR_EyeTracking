classdef tv_regularizer
    methods(Static)
        function [tv_loss, grad_tv] = compute_gradient(x)
            % Calculates Smoothed Isotropic Total Variation (TV) and its gradient.
            % Input 'x' must be a dlarray.
            
            % Circular shift to compute differences efficiently on GPU
            % dx = x(i+1, j) - x(i, j)
            dx = circshift(x, [-1, 0]) - x;
            
            % dy = x(i, j+1) - x(i, j)
            dy = circshift(x, [0, -1]) - x;
            
            % Smoothed L1 norm approximation
            epsilon = 1e-6; 
            grad_mag = sqrt(dx.^2 + dy.^2 + epsilon);
            
            % Sum of magnitudes
            tv_loss = sum(grad_mag, 'all');
            
            % Automatic differentiation w.r.t input x
            grad_tv = dlgradient(tv_loss, x);
        end
    end
end
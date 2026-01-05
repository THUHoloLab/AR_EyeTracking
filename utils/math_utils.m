classdef math_utils
    methods(Static)
        function L = power_iteration(all_OTFs, HR_weights, img_size)
            % Estimates the Lipschitz constant (spectral norm) of the system matrix A^T A
            % used for determining the optimal step size.
            
            iters = 20;
            if isa(all_OTFs, 'gpuArray')
                o_test = gpuArray.randn(img_size);
            else
                o_test = randn(img_size);
            end
            
            o_test = o_test / norm(o_test(:));
            L = 0;
            
            for i = 1:iters
                % Apply A then A^T
                m_sim = physics_model.forward(o_test, all_OTFs, HR_weights);
                o_adj = physics_model.adjoint(m_sim, all_OTFs, HR_weights);
                
                % Rayleigh quotient approximation
                L_new = norm(o_adj(:));
                o_test = o_adj / L_new;
                
                % Convergence check
                if i > 1 && abs(L_new - L)/L_new < 1e-4
                    L = L_new;
                    break;
                end
                L = L_new;
            end
            L = gather(L);
        end
    end
end
function [D1_out, D2_out, D3_out] = orthonormalize_frames(D1_in, D2_in, D3_in)
% Orthonormalizes the material frame {D1, D2, D3} at each point using
% modified Gram-Schmidt, assuming D3 is the primary direction.

N = size(D1_in, 2);
D1_out = zeros(3, N);
D2_out = zeros(3, N);
D3_out = zeros(3, N);

for k = 1:N
    d3 = D3_in(:, k);
    d1 = D1_in(:, k);

    % Normalize D3
    norm_d3 = norm(d3);
    if norm_d3 < eps
        % Handle potential zero vector - shouldn't happen in simulation
        % If it does, indicates a problem. Reinitialize to arbitrary frame?
        warning('D3 vector near zero at index %d during orthonormalization.', k);
        D3_out(:, k) = [0; 0; 1]; % Assign arbitrary direction
    else
        D3_out(:, k) = d3 / norm_d3;
    end

    % Project D1 onto plane normal to D3 and normalize
    d1_proj = d1 - dot(d1, D3_out(:, k)) * D3_out(:, k);
    norm_d1_proj = norm(d1_proj);
     if norm_d1_proj < eps
        % D1 was likely parallel to D3. Choose arbitrary orthogonal vector.
        % Find a vector not parallel to D3
        temp_vec = [1;0;0];
        if abs(dot(temp_vec, D3_out(:, k))) > 0.99
            temp_vec = [0;1;0];
        end
        d1_proj = cross(D3_out(:, k), temp_vec); % Create orthogonal vector
        d1_proj = d1_proj / norm(d1_proj); % Normalize it
        warning('D1 vector near parallel to D3 at index %d during orthonormalization.', k);
        D1_out(:, k) = d1_proj;
    else
        D1_out(:, k) = d1_proj / norm_d1_proj;
    end

    % Calculate D2 as the cross product to complete the right-handed system
    D2_out(:, k) = cross(D3_out(:, k), D1_out(:, k));

    % Optional check: Ensure D2 is unit norm (should be if D1/D3 are)
    % norm_d2 = norm(D2_out(:, k));
    % if abs(norm_d2 - 1) > 1e-6
    %    warning('D2 norm not 1 after orthonormalization at index %d.', k);
    %    D2_out(:, k) = D2_out(:, k) / norm_d2;
    % end

end

end


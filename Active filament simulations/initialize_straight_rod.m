function [X0, D10, D20, D30] = initialize_straight_rod(L, N, p_twist)
% Initializes a straight Kirchhoff rod along the z-axis with uniform twist p.

% Inputs:
%   L               : Total length of the rod (cm).
%   N               : Number of discretization points.
%   p_twist         : Total number of full twists along the length L.
%   perturb_amplitude : Amplitude of a sinusoidal perturbation in x (optional).
%   params          : (Not used for geometry, but kept for consistency)

% Outputs:
%   X0              : 3 x N matrix of initial vertex positions.
%   D10, D20, D30   : 3 x N matrices of the initial orthonormal material frame.

%ds = L / (N - 1); % Segment length for an open rod with N points

% --- Calculate Vertex Positions ---
s = linspace(0, L, N); % Arclength parameter
X0 = zeros(3, N);
X0(3, :) = s; % Straight line along z-axis

% --- Calculate Material Frame ---
% D3 should be approximately tangent. For a straight rod, it's z_hat.
% For perturbed, calculate tangent numerically.
dX = X0(:, 2:N) - X0(:, 1:N-1);
T = dX ./ vecnorm(dX, 2, 1);
% Average tangents at vertices
D30 = zeros(3, N);
D30(:, 2:N-1) = (T(:, 1:N-2) + T(:, 2:N-1)) / 2;
% Extrapolate ends (or use one-sided difference)
D30(:, 1) = T(:, 1);
D30(:, N) = T(:, N-1);
% Normalize D3
D30 = D30 ./ vecnorm(D30, 2, 1);

% Define D1 and D2 initially orthogonal to D3 (e.g., D1 along x_hat projected)
D1_guess = repmat([1; 0; 0], 1, N);
D10 = D1_guess - dot(D1_guess, D30, 1) .* D30;
D10 = D10 ./ vecnorm(D10, 2, 1);
% Handle cases where D3 is parallel to D1_guess (e.g., vertical rod part)
bad_idx = find(vecnorm(D10, 2, 1) < 1e-6);
if ~isempty(bad_idx)
    D2_guess_alt = repmat([0; 1; 0], 1, length(bad_idx));
    D10(:, bad_idx) = D2_guess_alt - dot(D2_guess_alt, D30(:,bad_idx), 1) .* D30(:,bad_idx);
    D10(:, bad_idx) = D10(:, bad_idx) ./ vecnorm(D10(:, bad_idx), 2, 1);
end
D20 = cross(D30, D10, 1);

% Add Twist
total_twist_angle = 2 * pi * p_twist;
alpha = (s / L) * total_twist_angle; % Linear twist distribution

cos_a = cos(alpha);
sin_a = sin(alpha);

% Rotate initial D1 and D2 around D3
D1_twisted = cos_a .* D10 + sin_a .* D20;
D2_twisted = -sin_a .* D10 + cos_a .* D20;

D10 = D1_twisted;
D20 = D2_twisted;

% --- Final Orthonormalization ---
[D10, D20, D30] = orthonormalize_frames(D10, D20, D30);

disp(['Initialized straight rod: L=', num2str(L), ', N=', num2str(N), ', p_twist=', num2str(p_twist)]);

end
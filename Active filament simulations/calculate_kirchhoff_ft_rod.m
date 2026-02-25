function [f_el, n_el] = calculate_kirchhoff_ft_rod(X, D1, D2, D3, params)
% Calculates discrete internal elastic force and torque densities for an OPEN Kirchhoff rod
% with boundary conditions appropriate for one CLAMPED end (k=1) and one FREE end (k=N).
% Based on Lim et al. (2008), Section 4, adapted for open loop & BCs.

% Inputs:
%   X       : 3 x N matrix of vertex positions.
%   D1, D2, D3 : 3 x N matrices of the material frame vectors.
%   params  : Structure containing material and discretization parameters:
%             .a1, .a2, .a3 : Bending/Twist moduli
%             .b1, .b2, .b3 : Shear/Stretch moduli
%             .ds      : Reference segment length

% Outputs:
%   f_el    : 3 x N matrix of elastic force density vectors at each vertex.
%   n_el    : 3 x N matrix of elastic torque density vectors at each vertex.

[~, N] = size(X);
if N < 2; error('Need at least 2 points (N>=2) for an open rod segment.'); end

% Extract parameters
a = [params.a1; params.a2; params.a3]; % Bending/Twist moduli
b = [params.b1; params.b2; params.b3]; % Shear/Stretch moduli
ds = params.ds;

% --- Initialization ---
% Midpoint values exist only for N-1 segments

%F_mid = zeros(3, N-1); % Internal forces at midpoints j+1/2 (j=1..N-1)
%N_mid = zeros(3, N-1); % Internal torques at midpoints j+1/2
D1mid = zeros(3, N-1);
D2mid = zeros(3, N-1);
D3mid = zeros(3, N-1);
dXds  = zeros(3, N-1); % Tangents at midpoints

% --- Loop over Midpoints (j+1/2) ---
% j runs from 1 to N-1
for j = 1:N-1
    jp1 = j + 1; % Index of the next vertex

    % --- Midpoint Geometry ---
    Dj = [D1(:,j), D2(:,j), D3(:,j)];
    Dj1 = [D1(:,jp1), D2(:,jp1), D3(:,jp1)];

    Aj = Dj1 * Dj'; % Rotation matrix from frame j to frame j+1

    % Ensure Aj is SO(3)
    [U, ~, V] = svd(Aj);
    Aj = U * V';
    if abs(det(Aj) - 1) > 1e-6
        % warning('Determinant of Aj is not 1 for segment %d. Det = %f', j, det(Aj));
    end

    % Midpoint frame: D_mid = sqrtm(A_j) * D_j
    try
        sqrtAj = sqrtm(Aj);
    catch ME
        warning('sqrtm failed for segment %d. Using axis-angle approx. Error: %s', j, ME.message);
        axis_angle = vrrotmat2vec(Aj);
        if ~any(isnan(axis_angle)) && norm(axis_angle(1:3)) > eps
             half_angle_rot_mat = vrrotvec2mat(axis_angle(1:3), axis_angle(4) / 2);
             sqrtAj = half_angle_rot_mat;
        else % Fallback if axis-angle is bad too (e.g. identity rotation)
            sqrtAj = eye(3);
        end
    end

    Dj_mid = sqrtAj * Dj;

    % Extract and store midpoint frame vectors (orthonormalize just in case)
    D3_midj = Dj_mid(:, 3) / norm(Dj_mid(:, 3));
    D1_midj = Dj_mid(:, 1);
    D1_midj = D1_midj - dot(D1_midj, D3_midj) * D3_midj;
    D1_midj = D1_midj / norm(D1_midj);
    D2_midj = cross(D3_midj, D1_midj);

    D1mid(:,j) = D1_midj;
    D2mid(:,j) = D2_midj;
    D3mid(:,j) = D3_midj;

    % Tangent vector at midpoint j+1/2
    dXds(:,j) = (X(:, jp1) - X(:, j)) / ds;
end

% --- Constitutive Relations at Midpoints ---
% Internal Forces F = F_k * D_k
F1_comp = b(1) * dot(dXds, D1mid);
F2_comp = b(2) * dot(dXds, D2mid);
F3_comp = b(3) * (dot(dXds, D3mid) - 1.0); % Stretch = dot(t, D3) - 1
F_mid = F1_comp .* D1mid + F2_comp .* D2mid + F3_comp .* D3mid;

% Finite difference for dD_k/ds
dD1_ds = (D1(:, 2:N) - D1(:, 1:N-1)) / ds;
dD2_ds = (D2(:, 2:N) - D2(:, 1:N-1)) / ds;
dD3_ds = (D3(:, 2:N) - D3(:, 1:N-1)) / ds;

% Darboux vector kappa = [kappa1; kappa2; kappa3] relative to midpoint frame
kappa1 = dot(dD2_ds, D3mid); % Component along D1_mid
kappa2 = dot(dD3_ds, D1mid); % Component along D2_mid
kappa3 = dot(dD1_ds, D2mid); % Component along D3_mid (Twist rate)


% Internal Torques N = N_k * D_k
N1_comp = a(1) * (kappa1);
N2_comp = a(2) * (kappa2);
N3_comp = a(3) * (kappa3);
N_mid = N1_comp .* D1mid + N2_comp .* D2mid + N3_comp .* D3mid;

% --- Calculate Force and Torque Densities at Vertices ---
f_el = zeros(3, N);
n_el = zeros(3, N);

% Interior points (k = 2 to N-1)
if N >= 3
    F_mid_diff_int = F_mid(:, 2:end) - F_mid(:, 1:end-1); % F_{k+1/2} - F_{k-1/2} for k=2..N-1
    N_mid_diff_int = N_mid(:, 2:end) - N_mid(:, 1:end-1); % N_{k+1/2} - N_{k-1/2} for k=2..N-1

    cross_term_int = cross(dXds(:, 2:end), F_mid(:, 2:end)) + cross(dXds(:, 1:end-1), F_mid(:, 1:end-1));     % cross(t_{k+1/2}, F_{k+1/2}) + cross(t_{k-1/2}, F_{k-1/2})

    f_el(:, 2:N-1) = - F_mid_diff_int / ds;
    n_el(:, 2:N-1) = - N_mid_diff_int / ds - 0.5 * (cross_term_int);
end

% Boundary point: free end (k=1)
% Assume F_{1-1/2} = 0, N_{1-1/2} = 0 for the finite difference
f_el(:, 1) = - F_mid(:, 1) / ds; % -(F_{1+1/2} - 0) / ds
n_el(:, 1) = - N_mid(:, 1) / ds - 0.5 * cross(dXds(:, 1), F_mid(:, 1)); % -(N_{1+1/2}-0)/ds - 0.5*(cross(t_{1+1/2},F_{1+1/2}) + cross(t_{1-1/2},0))

% Boundary point: Free end (k=N)
% Use F_{N+1/2} = 0, N_{N+1/2} = 0
if N >= 2 % Need at least one segment
    f_el(:, N) = F_mid(:, N-1) / ds; % -(0 - F_{N-1/2}) / ds
    n_el(:, N) = N_mid(:, N-1) / ds - 0.5 * cross(dXds(:, N-1), F_mid(:, N-1)); % -(0 - N_{N-1/2})/ds - 0.5*(cross(t_{N+1/2},0) + cross(t_{N-1/2},F_{N-1/2}))
end

end % End of function
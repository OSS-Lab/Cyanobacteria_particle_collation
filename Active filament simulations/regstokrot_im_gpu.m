function [u,w] = regstokrot_im_gpu(X_eval, x_0, f, n, epsilon, hwall, mu)
% Compute the velocity field and vorticity field (w = curl u) from regularized rotlets with the full image
% system for a no-slip wall at x = -hwall. 


% X_eval = gpuArray(X_eval);
% n = gpuArray(n);
% x_0 = gpuArray(x_0);

[~, M] = size(X_eval);
[~, N] = size(x_0);

% Handle edge cases
if N == 0 || M == 0
    u = zeros(3, M);
    return;
end

% --- Step 1: Reshape inputs for broadcasting ---
X_eval_r = reshape(X_eval, 3, 1, M);
x0_r = reshape(x_0, 3, N, 1);
n_r = reshape(n, 3, N, 1);
f_r = reshape(f, 3, N, 1);

% --- Step 2: Define image positions and geometric vectors ---
e1 = repmat([1; 0; 0],1,N,1); % Wall normal vector      % 3 x N x 1
hk_r = reshape(x_0(1,:) + hwall, 1, N, 1); % Distance from wall

% Image positions reflected across x=-hwall
x_im = x_0;
x_im(1,:) = -2 * hwall - x_0(1,:);
xim_r = reshape(x_im, 3, N, 1);

% Displacement vectors (size: 3 x N x M)
rk = X_eval_r - x0_r;
rkim = X_eval_r - xim_r;

% SQUARED vector norms (size: 1 x N x M)
r2 = sum(rk.^2, 1);
r2_im = sum(rkim.^2, 1);

%% --- Calculate strength vectors for the image systems ---

pk_r = cross(n_r, e1, 1); % Size: 3 x N x 1
nk_dot_e1 = sum(n_r .* e1, 1); % Size: 1 x N x 1
qk_r = n_r - nk_dot_e1 .* e1;

fk_dot_e1 = sum(f_r .* e1, 1);        % Size: 1 x N x 1
bk_r = 2 * fk_dot_e1 .* e1 - f_r;     % size: 3 x N x 1
mk_r = cross(f_r, e1, 1);             % size: 3 x N x 1


%% --- Compute helper functions ---
% size: 1 x N x M
Q_val_rk    = Q_psi(r2, epsilon);
Q_val_rkim  = Q_psi(r2_im, epsilon);
D1_phi_rkim  = D1_phi(r2_im, epsilon);
D2_phi_rkim  = D2_phi(r2_im, epsilon);
H3_rkim      = H3(r2_im, epsilon);
H4_rkim      = H4(r2_im, epsilon);
D1_psi_rkim  = D1_psi(r2_im, epsilon);
D2_psi_rkim  = D2_psi(r2_im, epsilon);
D1_psi_rk   = D1_psi(r2, epsilon);
D2_psi_rk   = D2_psi(r2, epsilon);
H5_rkim     = H5(r2_im, epsilon);

H1_rk       = H1_psi(r2, epsilon);
H2_rk       = H2_psi(r2, epsilon);
H1_rkim     = H1_psi(r2_im, epsilon);
H2_rkim     = H2_psi(r2_im, epsilon);
Q_psi_rk    = Q_psi(r2, epsilon);
Q_psi_rkim  = Q_psi(r2_im, epsilon);


%% ---  Velocity terms ---
% All intermediate terms are 3 x N x M

% Term 1: Source and Image Rotlets
n_r_exp = repmat(n_r,1,1,M);
cross_nk_rk = cross(n_r_exp, rk, 1);
cross_nk_rkim = cross(n_r_exp, rkim, 1);
term0 =  0.5 * Q_val_rk  .* cross_nk_rk;
term1 = -0.5* Q_val_rkim .* cross_nk_rkim;

% Term 2: Potential Dipole
pk_dot_rkim = sum(pk_r .* rkim, 1); % Size: 1 x N x M
term2 = hk_r .* (pk_r .* D1_phi_rkim + pk_dot_rkim .* rkim .* D2_phi_rkim);

% Term 3: Stokeslet Doublet part 1
e1_dot_rkim = rkim(1,:,:); % Size: 1 x N x M
term3_partA = (pk_dot_rkim .* e1 + e1_dot_rkim .* pk_r) .* H3_rkim;
term3_partB = (e1_dot_rkim .* pk_dot_rkim .* rkim) .* D2_phi_rkim;
term3 = -(term3_partA + term3_partB);

% Term 4: Stokeslet Doublet part 2
term4 = -hk_r .* (pk_r .* H3_rkim + e1_dot_rkim .* cross_nk_rkim .* H4_rkim);

% Term 5: Rotlet Doublet and Quadrupole parts
qk_r_exp = repmat(qk_r,1,1,M);
cross_qk_rkim = cross(qk_r_exp, rkim, 1);
term5 = - H3_rkim .* cross_qk_rkim + hk_r.^2 .* H4_rkim .* cross_nk_rkim;

% --- Step 6: Sum all terms and finalize ---
u_rot = term0 + term1 + term2 + term3 + term4 + term5;



%% --- Velocity formula ---
% All intermediate terms will be tensors of size 3 x N x M

% Term 1: Source Stokeslet
fk_dot_rk = sum(f_r .* rk, 1);
term1 = f_r .* H1_rk + fk_dot_rk .* rk .* H2_rk;

% Term 2: Image Stokeslet
fk_dot_rkim = sum(f_r .* rkim, 1);
term2 = - (f_r .* H1_rkim + fk_dot_rkim .* rkim .* H2_rkim);

% Term 3: Image Potential Dipole
bk_dot_rkim = sum(bk_r .* rkim, 1);
term3 = (hk_r.^2) .* ( -bk_r .* D1_phi_rkim - bk_dot_rkim .* rkim .* D2_phi_rkim );

% Term 4 & 5: Image Stokes Doublet (combined from Park et al. formula)
bk_dot_e1 = sum(bk_r .* e1, 1);      % 1xNx1
e1_dot_rkim = rkim(1,:,:);           % 1xNxM
term4_partA = (bk_dot_e1 .* rkim + e1_dot_rkim .* bk_r) .* H2_rkim;
term4_partB = (bk_dot_rkim .* e1) .* (H3_rkim - H2_rkim) ;
term4 = 2 * hk_r .* (term4_partA + term4_partB);
term5 = hk_r .* (e1_dot_rkim .* bk_dot_rkim .* rkim) .* D2_phi_rkim;

% Term 6: Rotlet-like term from the Stokes Doublet
mk_r_exp = repmat(mk_r,1,1,M);
cross_mk_rkim = cross(mk_r_exp, rkim, 1);
term6 = 2 * hk_r .* H3_rkim .* cross_mk_rkim;


%% --- Final velocity ---
% Sum all 3xNxM tensors
u_tensor = u_rot + term1 + term2 + term3 + term4 + term5 + term6;
%u_tensor =   term1; disp('no image'); 

% Sum contributions from all N sources for each M evaluation point
u_summed = sum(u_tensor, 2); % Result is 3 x 1 x M
u = squeeze(u_summed) / mu;

% Handle the case of M=1, where squeeze might create a row vector
if M == 1
    u = u(:);
end




%% --- Vorticity formula (w = curl(u)) ---

%% Rotlets
% Term 1: Source Rotlet (1/2 curl R_psi)
nk_dot_rk = sum(n_r .* rk, 1);
w_term1 = - 0.25 * (D1_psi_rk .* n_r + D2_psi_rk .* nk_dot_rk .* rk);

% Term 2: Image Rotlet (-1/2 curl R_phi)
nk_dot_rkim = sum(n_r .* rkim, 1);
w_term2 = 0.25 * (D1_psi_rkim .* n_r + D2_psi_rkim .* nk_dot_rkim .* rkim);

% Term 3: Doublet D (-1/2 curl (D[p,e1]+D[e1,p]))
pk_dot_rkim = sum(pk_r .* rkim, 1);
e1_dot_rkim = rkim(1,:,:);
cross_pk_rkim = cross(repmat(pk_r,1,1,M), rkim, 1);
cross_e1_rkim = cross(repmat(e1,1,1,M), rkim, 1);
w_term3 = -0.5 * (D2_phi_rkim - H4_rkim) .* (e1_dot_rkim .* cross_pk_rkim + pk_dot_rkim .* cross_e1_rkim);


% Term 4: Potential Dipole P (-h/2 curl P[p])
w_term4 = -hk_r .* H4_rkim .* cross_pk_rkim;

% Term 5: Rotlet Difference R (1/2 curl (R_psi[q] - R_phi[q]) = 1/2 curl (H3(q x r)))
qk_dot_rkim = sum(qk_r .* rkim, 1);
w_term5 = - 0.5 * (r2_im .* H4_rkim + 2 * H3_rkim) .* qk_r + (0.5 * H4_rkim .* qk_dot_rkim) .* rkim;

% Term 6: Rotlet Doublet J (h/2 curl (J_psi - J_phi))
term6a = e1_dot_rkim .* (r2_im .* H5_rkim + 3*H4_rkim ).* n_r;
scalar_term6b = - nk_dot_e1 .* H4_rkim - e1_dot_rkim .* nk_dot_rkim .* H5_rkim;
w_term6 = - 0.5 * hk_r .*( term6a + scalar_term6b .* rkim - H4_rkim .* cross_pk_rkim );

% Term 7: Quadrupole Q (-h^2/4 curl Q[n])
w_term7 = 0.5 * hk_r.^2 .*( (2*H4_rkim + r2_im .* H5_rkim) .* n_r - H5_rkim .* nk_dot_rkim .* rkim ) ;

% --- Final vorticity ---
w_rot = 2*(w_term1 + w_term2 + w_term3 + w_term4 + w_term5 + w_term6 + w_term7);

%% Stokeslets 
% Term 1 & 2: Source and Image Stokeslet
f_r_exp = repmat(f_r,1,1,M);
w_term1 = 0.5 * Q_psi_rk .* cross(f_r_exp, rk, 1);
w_term2 = -0.5 * Q_psi_rkim .* cross(f_r_exp, rkim, 1);

% Term 3: Image Potential Dipole
w_term3 = (hk_r.^2) .* H4_rkim .* cross(repmat(bk_r,1,1,M), rkim, 1);

% Term 4: Image Stokes Doublet
bk_dot_rkim = sum(bk_r .* rkim, 1);
% This term is h * 1/2 * curl(D[e1,b]+D[b,e1]), simplified from appendix
w_term4 = hk_r .* (D2_phi_rkim - H4_rkim) .* (bk_dot_rkim .* cross(repmat(e1,1,1,M), rkim, 1));

% Term 5 and 6: Rotlet-like term from the Stokes Doublet 
w_term5 = hk_r.*Q_psi_rkim.*mk_r; 

mk_dot_rkim = sum(mk_r .* rkim, 1);
w_term6 = hk_r .* ( (r2_im .* H4_rkim + 2 * H3_rkim) .* mk_r - H4_rkim .* mk_dot_rkim .* rkim); 


%% --- Final vorticity (curl u) ---
w_tensor = w_rot + 2*(w_term1 + w_term2 + w_term3 + w_term4 + w_term5 + w_term6);


w_summed = sum(w_tensor, 2); % Sum over N -> 3 x 1 x M
w = squeeze(w_summed) / mu;

if M == 1
    w = w(:);
end



% u = gather(u); 
% w = gather(w); 

end




%% --- Helper functions for the image system

function val = Q_psi(r2, epsilon) % Eq. 33
    e2 = epsilon^2;
    val = (5*e2 + 2*r2) ./ (8 * pi * (r2 + e2).^(5/2));
end

function val = D1_psi(r_sq, epsilon) % Eq. A22
    e2 = epsilon^2; val = (-10*e2^2 + 7*r_sq.*e2 + 2*r_sq.^2) ./ (8 * pi * (r_sq + e2).^(7/2));
end

function val = D2_psi(r2, epsilon) % Eq. 35
    e2 = epsilon^2;
    val = (-21*e2 - 6*r2) ./ (8 * pi * (r2 + e2).^(7/2));
end

function val = D1_phi(r2, epsilon) % Eq. 36
    e2 = epsilon^2;
    val = (-2*e2 + r2) ./ (4 * pi * (r2 + e2).^(5/2));
end

function val = D2_phi(r2, epsilon) % Eq. 37
    e2 = epsilon^2;
    val = -3 ./ (4 * pi * (r2 + e2).^(5/2));
end

function val = H1_psi(r2, epsilon) % Eq. 31
    e2 = epsilon^2;
    val = (2*e2 + r2) ./ (8 * pi * (r2 + e2).^(3/2));
end

function val = H2_psi(r2, epsilon) % Eq. 32
    e2 = epsilon^2;
    val = 1 ./ (8 * pi * (r2 + e2).^(3/2));
end

function val = H3(r2, epsilon) % Eq. 38 (H3 = G_phi'/r + H2)
    e2 = epsilon^2;
    val =  -3*e2 ./ (8 * pi * (r2 + e2).^(5/2));
end

function val = H4(r2, epsilon) % Eq. 39 / A21
    e2 = epsilon^2;
    val = 15*e2 ./ (8 * pi * (r2 + e2).^(7/2));
end

function val = H5(r_sq, epsilon) % Eq. 40 / A31
    e2 = epsilon^2; val = -105*e2 ./ (8 * pi * (r_sq + e2).^(9/2));
end





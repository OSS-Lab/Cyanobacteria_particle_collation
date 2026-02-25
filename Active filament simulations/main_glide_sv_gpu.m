function [Xf_save,D1_save,cellprop_save] = main_glide_sv_gpu(params,cellprop,filename)

Nf = params.Nf;
ds = params.ds;

%% --- Loading parameters --------------------------------------------------------------------------------------------
mu = params.mu; % 0.01e-4;         % Fluid viscosity
epsilon = params.epsilon ;
Crep = params.Crep;      % Repulsion stiffness (adjust based on units/stability)
Dmrep = params.Dmrep;     % Interaction distance
T_final = params.T_final;         % Final simulation time (s)
dt = params.dt;              % 1e-8;  % Time step (s)
save_interval = params.save_interval;


%% --- Initialization --------------------------------------------------------------------------------------------
[Xf, D1, D2, D3] = initialize_straight_rod(params.Lf, params.Nf, 0);
Xf = Xf - mean(Xf,2);

% Noise in initial configuration
Xf = Xf + params.Lf/1e5.*rand(size(Xf));

%% --- Time Stepping --------------------------------------------------------------------------------------------
num_steps = round(T_final / dt);

Xf_save = zeros(3,Nf,floor(num_steps/save_interval));
D1_save = zeros(3,Nf,floor(num_steps/save_interval));
cellprop_save = zeros(1,params.Ncells,floor(num_steps/save_interval));

disp(['Starting time stepping for ', num2str(T_final), 's with dt=', num2str(dt), ' and ',num2str(num_steps,'%.2e'),' steps']);
disp([num2str(size(Xf,2)), ' filament points']);
%load('cmbwr.mat','cm');

tic;

t0 = 1;

for i = t0:num_steps
    t = i * dt;

    % ------------------------ Compute all the forces and torques -----------------------------------------------------------------------------------------
    % Calculate flow from filament elasticity
    [f_el, n_el] = calculate_kirchhoff_ft_rod(Xf, D1, D2, D3, params);

    % Surface Interaction

    % Cells to point actuation
    forwardback = cos(cellprop);
    [fsurf,nsurf] = gliding_force_torque(Xf, Crep, params.Cadh, Dmrep, params.fglide, params.nglide, forwardback);

    % Force/Torque density -> Force/Torque per segment for Stokeslets
    g_ffl = - f_el * ds - fsurf*ds ; % Force applied by filament on fluid
    n_ffl = - n_el * ds - nsurf*ds;  % Torque applied by filament on fluid

    % ------------------------ Update the position of filament and the orientation of the triads ---------------------------------------------------------
    % Calculate Fluid Velocity and Angular Velocity at Filament Points

    [U, W] = regstokrot_im_gpu(Xf, Xf, g_ffl, n_ffl, epsilon, params.film_sz, mu);
    W = W/2;

    % Update filament Position (Explicit Euler)
    Xf = Xf + dt * U;

    % Update Flagellar Triads Orientation (Rodrigues' formula)
    angle = vecnorm(W, 2, 1) * dt;
    axis_rot = W ./ vecnorm(W, 2, 1);
    axis_rot(:, angle < eps) = repmat([1; 0; 0], 1, sum(angle < eps));
    angle(angle < eps) = 0;

    cos_a = cos(angle); sin_a = sin(angle); om_cos_a = 1 - cos_a;
    vec_to_cross_mat = @(v) [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
    D1_new = zeros(3, Nf); D2_new = zeros(3, Nf); D3_new = zeros(3, Nf);
    for k = 1:Nf
        ax = axis_rot(:, k);
        ax_cross = vec_to_cross_mat(ax);
        R = eye(3) * cos_a(k) + om_cos_a(k) * (ax * ax') + sin_a(k) * ax_cross;
        D1_new(:,k) = R * D1(:,k);
        D2_new(:,k) = R * D2(:,k);
        D3_new(:,k) = R * D3(:,k);
    end
    D1 = D1_new; D2 = D2_new; D3 = D3_new;

    % Orthonormalize Frames
    [D1, D2, D3] = orthonormalize_frames(D1, D2, D3);

    % ------------------------ Save results --------------------------------------------------------------------------------------------------------
    if mod(i-1, save_interval) == 0
        Xf_save(:,:,(i-1)/save_interval+1) = Xf;
        cellprop_save(:,:,(i-1)/save_interval+1) = cellprop;
        D1_save(:,:,(i-1)/save_interval+1) = D1;
        inow = i; 
        toc
        save(filename,"Xf_save","D1_save","cellprop_save","params","inow","Xf", "D1", "D2", "D3","cellprop");
    end

    % ------------------------ Check for instability ------------------------------------------------------------------------------------------------------
    if any(isnan(Xf(:))) || any(isinf(Xf(:))) || any(isnan(D1(:))) || any(isnan(D2(:))) || any(isnan(D3(:)))
        error('Simulation became unstable at t = %f s. Reduce dt.', t);
    end

end
toc;

end
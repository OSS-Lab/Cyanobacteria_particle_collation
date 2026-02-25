function [filename] = run_glideflex
% Main function for the elastohydroodynamic simulations of an active
% filament gliding on a surface


%% ----------------------------------------------------------------------------------------------------
%% PARAMETERS -----------------------------------------------------------------------------------------

% The typical aprameters to be tuned are 
% Ncells (filament length) [range 100 - 1000]
% flex (filament stiffness, decrease to get a flexible filament) [range 0.01 - 1]
% Gamma (twist to bend stiffness ratio) [range 0.1 - 1 - 10]
% coeffadh (adhesions strength) [range 0 - 5 - 10]

%% Cell filament parameters -----------------------------------------------------------------------------------------
Ncells = 50;               % Number of filament points; if (if ds = 1 as in default), this is the filament length
ds = 1;                    % Size of the mesh (distance between points) 
cellsize = 1;              % number of neighbouring points that are connected (default 1)
Nf = Ncells*cellsize;      % Number of mesh points
Lf = ds*(Nf-1);            % Length of filament

%% Filament mechanics - Kirchhoff Rod -----------------------------------------------------------------------------------------
flex = 0.01;                % Relative stiffness, denoted K in manuscript. flex = 1 for physical cyano (Fluctiforma draycotensis) 
Gamma = .1;

params.a1 = flex*1e5;       % Bending modulus  K_b  (nN.µm^2)
params.a2 = flex*1e5;       % Bending modulus  K_b
params.a3 = Gamma*params.a2;  % Twisting modulus  K_t

params.b1 = flex*1e6;        % Shear modulus
params.b2 = flex*1e6;        % Shear modulus
params.b3 = flex*1e6;        % Stretching modulus

%% Propulsion strength -----------------------------------------------------------------------------------------
ncoeff = 1*12.5;             % Ratio of gliding moments to gliding forces (12.5 to match flucti propulsion)

params.fglide = 1*ds;        % Force when in contact with surface (gliding propulsion) (force density 1 nN/µm)
params.nglide = ncoeff*params.fglide;        % Torque when in contact with the surface (gliding rotation)

%% Surface Interaction -----------------------------------------------------------------------------------------
coeffadh = 20;                 % Strength of adhesion with the surface relative to gliding forces (default 5)
% Repulsion force between body and surface
Crep = 1e3;      % Repulsion stiffness (adjust based on units/stability)
Dmrep = 0.1;     % Steric interaction distance
params.Cadh = coeffadh*params.fglide;          % Adhesion force to the surface (normal to surface)

%% Fluid properties -----------------------------------------------------------------------------------------
params.mu = .2;               % water 1e-6, slime up to 1e-3 (s.nN.µm^-2)
params.film_sz = 1+3*ds;      % thickness of gliding film in µm (add 1 radius 3 ds)

%% Numerical parameters -----------------------------------------------------------------------------------------
params.T_final = 10;        % duration of a simulation
params.dt = .00001/flex; %  .0000125/flex;
params.save_interval = 25000 * flex;
% Regularized Stokeslet/Rotlet Parameters
epsilon = 3*ds ;
 

% Save all parameters in the params structure
params.cellsize = cellsize;
params.Ncells = Ncells;
params.Nf = Nf;
params.Ncells = Ncells;
params.Lf = Lf;
params.ds = ds;
params.flex = flex; 
params.Crep = Crep;      
params.Dmrep = Dmrep;    





% Numerical parameters -----------------------------------------------------------------------------------------
T_final = params.T_final;         % Final simulation time (s)
dt = params.dt;              % 1e-8;  % Time step (s)
save_interval = params.save_interval;

% Saving parameters for current sim
params.epsilon = epsilon;





%% -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
%% INITIAL CONDITIONS -----------------------------------------------------------------------------------------
cellprop_ini = pi*ones(1,Ncells);   %Forward propulsion
% cellprop_ini(1:floor(Ncells/2)) = 0;          %Uncomment for mid-filament reversal



%% -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

% The results will be saved in filename (as .mat files)
t = datetime; t.Format = 'MM-dd-hh-mm';
filename = append("glide_revmid_N",num2str(Ncells),"-K",num2str(flex),"-Gamma",num2str(Gamma),"-n",num2str(ncoeff),"-cadh",num2str(coeffadh),"_",string(t),".mat");

tic
[Xf_save,D1_save,cellprop_save] = main_glide_sv_gpu(params,cellprop_ini,filename);
toc

save(filename,"Xf_save","D1_save","cellprop_save","params");

end
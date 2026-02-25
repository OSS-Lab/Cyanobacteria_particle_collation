## Active Filament Gliding Simulations

### Overview
Source code for the Biophysical model reported "Emergent particle collection by cyanobacteria underpinned by motility and filament buckling"

Author: Albane Thery

### Running a Simulation
Launch a simulation from MATLAB by running: filename = run_glideflex;
Parameters can be adjusted directly in the file, in particular:
•	Ncells: number of cells (filament length)
•	flex: dimensionless stiffness scaling (K)
•	Gamma: twist-to-bend stiffness ratio
•	coeffadh: adhesion scaling
•	T_final: total simulation time

###  Visualization
Plot the results of the simulation using: vis_filament(filename,0)
Example command vis_filament('glide_revmid_N1000-K1-Gamma1-n12.5-cadh5_example.mat', 0)
Set the second parameter (make_movie) to 1 to save them as an filename.avi file and 0 otherwise.

###  Parameters
Key parameters to adjust include:
- Ncells: number of cells (filament length)
- flex: dimensionless stiffness scaling
- Gamma: twist-to-bend stiffness ratio
- coeffadh: adhesion scaling
- dt: numerical time step
- T_final: total simulation time
These are set near the top of run_glideflex.m.

###  Output Format
The saved data fields include:
- Xf_save: filament centerline coordinates over time (3 × Nf × Nt)
- D1_save: first director frame (3 × Nf × Nt) (partial information on the filament orientation, for plotting)
- cellprop_save: propulsion state at each timestep 
- params: full parameter structure
- inow: last completed timestep
- Xf, D1, D2, D3: current configuration (full information)
Here, Nt is the number of saved frames. Because the simulations are long, we ensure that they are regularly saved and the results can be plotted and analysed even when the simulations haven’t run fully. 


### Repository Contents
#### For simulations 
- run_glideflex.m
Main entry point. Defines parameters, initializes the filament, and calls the solver.
- main_glide_sv_gpu.m
Core solver. Implements force and torque computations, hydrodynamic mobility evaluation, rod orientation updates, and time stepping.
- Supporting functions for the simulations (must be in MATLAB path):
  - initialize_straight_rod (initialise the straight filaments with given mechanical properties)
  - calculate_kirchhoff_ft_rod (computes the Kirchoff forces and torques on the filament for a given configuration)
  - gliding_force_torque (forces and torques from the gliding)
  - regstokrot_im_gpu (computes the fluid flow from the filament configuration and the active and Kirchoff forces and torques, using regularised stokeslets and rotlets and their images.)
(this can be accelerated by uncommenting the gpu commands at the start and finish of the function on compatible computers)
  - orthonormalize_frames (ensure that the triads are orthonormal after each rotation) 

#### For visualisation 
- vis_filament.m
- Supporting functions for plotting (must be in MATLAB path):
  - plot_cyano_snap (plots the filament at a given time point. Lighting, colours etc can be edited in this function) 

### Note on numerical stability
- Numerical instability results in NaN or Inf values; reduce the timestep dt if this occurs. 
More rigid filaments may require a smaller dt (this is built in through flex). Generally these simulations are very stiff and require small timesteps. 



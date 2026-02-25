function [] = vis_filament(filename, make_movie)
% Plots the results of the simulation in filename 
% If make_movie is 1, saves a movie as filename.avi

load(filename,"Xf_save","params","D1_save","cellprop_save","inow");

if ~exist('inow','var'), inow = 100*size(Xf_save,3)*params.save_interval; end

tint = 1;       % adjust the interval between plotted frames
if make_movie, F(floor(min(size(Xf_save,3),floor(inow/params.save_interval))/tint)) = struct('cdata',[],'colormap',[]); end

%% Adjust figure position and axis limits
%fig = gcf; fig.Position = [900 200 900 300];
boxsize = [min(Xf_save(3,:,:),[],"all"),max(Xf_save(3,:,:),[],"all")];
boxwidth = [min(-50,min(Xf_save(2,:,:),[],"all")),max(50,max(Xf_save(2,:,:),[],"all"))];
boxheight = [-5,max(20,max(Xf_save(1,:,:),[],"all"))];

%% Timestepping 
for t =  1:tint:min(size(Xf_save,3),floor(inow/params.save_interval)) % Run until current point in simulation
    clf
    cellprop = cellprop_save(:,:,t);
    forwardback = cos(cellprop);
    plot_cyano_snap(Xf_save(:,:,t),D1_save(:,:,t),forwardback,params,boxsize,boxwidth,boxheight);
    title(['t = ',num2str(t*params.dt*params.save_interval,'%.0f'),' s'],'FontSize', 18, 'Units', 'normalized', 'Position', [0.2, 0.7, 0]); 
    
    ax = gca;
    ax.Color = 'none';
    % ax.XColor = 'none'; ax.YColor = 'none'; ax.ZColor = 'none';

    drawnow limitrate

    if make_movie, F(1+(t-1)/tint) = getframe(gcf); end
end

if make_movie
    moviename = replace(filename, '.mat', '.avi');
    v = VideoWriter(moviename);
    v.FrameRate= 10;
    v.Quality= 100;
    open(v)
    writeVideo(v,F);
    close(v)
end


end
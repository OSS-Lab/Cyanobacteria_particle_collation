function [] = plot_cyano_snap(Xf,D1,forwardback,params,boxsize,boxwidth,boxheight,f_color)


rfil = 1*1.5*params.ds; % visual radius of filament - increase to get thicker filament, more visible in main text figures

n_circ = 15;
if nargin < 8 
f_color = [0.05,0.54,0.21]; % [0.99,0.24,0.38];  %color of the filament 
end
edge_alpha = 0;

[~, Nf] = size(Xf);
Lf = params.Lf;


% % % Parameters for image % % % -------------------------------------------------------------------------------------------------------------------
hold on;
xlim(1.2*boxwidth); ylim(1.2*boxsize); zlim(1.2*boxheight);
%zlim([min(-2*params.film_sz,-Lf/20) max(4,Lf/3)]);
view(-50,15);
daspect([1 1 1]);

ax = gca;
% % Add camlight 
if isempty(findobj(ax, 'Type', 'light'))
    camlight(ax, 'headlight');
end
material(ax, 'shiny');
grid(ax, 'off');
ax.Position = [0.05,0.05,0.9,0.9];


% Uncomment to plot side points - visualise orientation, propulsion strength and direction % % % -----------------
% load('cmbwr.mat','cm'); % colormap from red to blue for propulsion pattern 
% for i = 1:Nf
%     col = 1 + floor( (tanh(forwardback(i))+1) .*50);
%     plot3(Xf(2,i)+ 1.02*rfil.* D1(2,i), Xf(3,i)+ 1.02*rfil.* D1(3,i), Xf(1,i)+ 1.02*rfil.* D1(1,i),'.','MarkerSize',6,'Color',cm(col,:) );
% end


%% % Plot crawling surface -------------------------------------------------------------------------------------------------------------------
h = surf(2*[-Lf -Lf; Lf Lf],2*[boxsize(1) boxsize(2); boxsize(1) boxsize(2)],[0 0; 0 0]-3*params.ds,'FaceAlpha',.25,'FaceColor',[0.50,0.50,0.50],'EdgeColor','none');
% surf(2*[-Lf -Lf; Lf Lf],2*[boxsize(1) boxsize(2); boxsize(1) boxsize(2)],-[params.film_sz params.film_sz; params.film_sz params.film_sz],'FaceAlpha',.25,'FaceColor',[0.50,0.50,0.50],'EdgeColor','none');
h.AmbientStrength = 0.9;

%% Plot scale bar 100 µm
plot(-[50 50], [Lf/2, Lf/2 - 100],'k','LineWidth',1);


%% Plot the filament body -------------------------------------------------------------------------------------------------------------------
% --- Calculate Tangents and Normal Frames ---
T = zeros(3, Nf); % Tangent vectors
N1 = zeros(3, Nf); % First normal vector
N2 = zeros(3, Nf); % Second normal vector

% Calculate segment tangents (forward difference)
T_segments = Xf(:, 2:end) - Xf(:, 1:end-1);
T_segments = T_segments ./ vecnorm(T_segments, 2, 1); % Normalize

% Estimate tangents at vertices by averaging segment tangents
T(:, 2:Nf-1) = (T_segments(:, 1:end-1) + T_segments(:, 2:end)) / 2;
% Handle ends (use segment tangent)
T(:, 1) = T_segments(:, 1);
T(:, Nf) = T_segments(:, end);
% Normalize vertex tangents
T = T ./ vecnorm(T, 2, 1);

% Create normal vectors (N1, N2) orthogonal to T
% Use a robust method to find the first normal (avoids issues if T is vertical)
vec_ref = [0; 0; 1]; % Reference vector
if abs(abs(dot(T(:,1), vec_ref)) - 1) < 1e-6 % Handle case where first tangent is vertical
    vec_ref = [1; 0; 0]; % Use a different reference
end
N1(:,1) = cross(T(:,1), vec_ref);
N1(:,1) = N1(:,1) / norm(N1(:,1));
N2(:,1) = cross(T(:,1), N1(:,1)); % N2 = T x N1

% Propagate the frame along the curve using rotation minimizing approach (approx)
% Rotate the previous frame {N1(i-1), N2(i-1)} to align T(i-1) with T(i)
for i = 2:Nf
    T_prev = T(:, i-1);
    T_curr = T(:, i);

    axis_rot = cross(T_prev, T_curr);
    angle_rot = acos(dot(T_prev, T_curr)); % Angle between tangents

    if norm(axis_rot) > 1e-8 && abs(angle_rot) > 1e-6 % Only rotate if tangents are different
        axis_rot = axis_rot / norm(axis_rot);
        % Use Rodrigues' rotation formula to rotate N1_prev and N2_prev
        % R = I + sin(a)*K + (1-cos(a))*K^2 where K is skew-symm matrix of axis
        K = [0 -axis_rot(3) axis_rot(2); axis_rot(3) 0 -axis_rot(1); -axis_rot(2) axis_rot(1) 0];
        R = eye(3) + sin(angle_rot)*K + (1-cos(angle_rot))*(K*K);

        N1(:, i) = R * N1(:, i-1);
        N2(:, i) = R * N2(:, i-1);
    else % Tangents are parallel, keep the same frame orientation
        N1(:, i) = N1(:, i-1);
        N2(:, i) = N2(:, i-1);
    end
end

% --- Generate Cylinder Surface Mesh ---
theta = linspace(0, 2*pi, n_circ + 1); % Angles around the circle (+1 to close)
cos_th = cos(theta);
sin_th = sin(theta);

% Preallocate surface coordinate matrices
X_surf = zeros(n_circ + 1, Nf);
Y_surf = zeros(n_circ + 1, Nf);
Z_surf = zeros(n_circ + 1, Nf);

% Create points for each circular cross-section
for i = 1:Nf
    % Points = Center + radius * (cos(theta)*N1 + sin(theta)*N2)
    points_on_circle = Xf(:, i) + rfil * (N1(:, i) * cos_th + N2(:, i) * sin_th);
    X_surf(:, i) = points_on_circle(2, :)';
    Y_surf(:, i) = points_on_circle(3, :)';
    Z_surf(:, i) = points_on_circle(1, :)';
end

h = surf( X_surf, Y_surf, Z_surf);


% --- Set Appearance and lighting ---------------------------------------------------------------------------------------------------------------------------------------------------------------
h.FaceColor = f_color;
h.EdgeColor = 'none';
h.EdgeAlpha = edge_alpha; % Control edge transparency
h.FaceLighting = 'gouraud'; % Smooth lighting
h.AmbientStrength = 0.6;
h.DiffuseStrength = 0.8;
h.SpecularStrength = 0.8;
h.SpecularExponent = 5;
h.BackFaceLighting = 'reverselit';


% Plot head and tail extremities
[X, Y, Z] = sphere(n_circ);
X = rfil*X + Xf(2,1);
Y = rfil*Y + Xf(3,1);
Z = rfil*Z + Xf(1,1);
surf(X, Y, Z,'FaceColor',f_color,'EdgeColor','none','AmbientStrength',.6,'DiffuseStrength',.8,'SpecularStrength',.8,'SpecularExponent',5,'FaceLighting','gouraud','BackFaceLighting','reverselit');
[Z, Y, X] = sphere(n_circ);
X = rfil*X + Xf(2,end);
Y = rfil*Y + Xf(3,end);
Z = rfil*Z + Xf(1,end);
surf(X, Y, Z,'FaceColor',f_color,'EdgeColor','none','AmbientStrength',.6,'DiffuseStrength',.8,'SpecularStrength',.8,'SpecularExponent',5,'FaceLighting','gouraud','BackFaceLighting','reverselit');

end


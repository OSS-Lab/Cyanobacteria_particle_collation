function [fsurf,nsurf] = gliding_force_torque(Xf, Crep, Cadh, Dsurf, fglide, nglide,forwardback)

[~, Nf] = size(Xf);

Xfdir = [Xf(:,1),Xf(:,1:end-1)]-[Xf(:,2:end),Xf(:,end)]; 
Xfdir(1,:) = 0;  % Gliding is parallel to the surface 
Xfdir = Xfdir./repmat(vecnorm(Xfdir,2,1),[3,1]); 

is_near_surface = Xf(1,:) < Dsurf ;

% Gliding force 

fsurf = fglide*Xfdir.*repmat(forwardback.*is_near_surface,3,1); 
nsurf = nglide*Xfdir.*repmat(forwardback.*is_near_surface,3,1); 

% Steric repulsion 
is_contact_surface = Xf(1,:) < 0 ;
fsteric = Crep.* Xf(1,:).*[is_contact_surface; zeros(2,Nf)]; 

% Adhesion 
dsurface = Dsurf/2; 
is_sticky_zone = ( Xf(1,:)>dsurface ) .* ( Xf(1,:)<2*dsurface ) ;
fadhesion = Cadh.*  (1+cos( (Xf(1,:)-3*dsurface/2)/dsurface*2*pi))/2 .*[is_sticky_zone; zeros(2,Nf)]; 

fsurf = fsteric + fsurf + fadhesion; 

end
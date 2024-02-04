%Aviv 2021-12-15
%Create Atlantic etc masks for ECCO4
% addpath 'C:/Users/user/Downloads' %where we have calcalphabeta, dens and sw_cp
clear all
OS = 'Linux';%'Windows';
ECCOv = 'V4r3';%'V4r4';%
if strcmp(ECCOv,'V4r4')
    addpath ./ECCOv4r4Andrew; 
    dirv4r4 = fnos('D:/Research/NumericalModels/ECCO/Version4/Release4/',OS); 
else
    dirv4r3 = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/',OS); 
end
p = genpath(fnos('D:/Aviv/Research/MATLAB_Scripts/Ocean/gcmfaces/',OS)); addpath(p);
if strcmp(ECCOv,'V4r3')
    dirGrid = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/nctiles_grid/',OS);
    OutputFolder = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/',OS);
else
%     dirGrid =
%     fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/nctiles_grid/',OS);
%     %Is this right?
    OutputFolder = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release4/',OS);
end


%% "global" loads
nFaces = 5; %nFaces is the number of faces in this gcm set-up of current interest.
fileFormat = 'nctiles';%'compact'; %fileFormat is the file format ('straight','cube','compact')
memoryLimit=0; omitNativeGrid=isempty(dir([dirGrid 'tile001.mitgrid']));
grid_load(dirGrid,nFaces,fileFormat,memoryLimit,omitNativeGrid); %Load the grid.
gcmfaces_global; % Define global variables

%%
[X,Y,B]=convert2pcol(mygrid.XC,mygrid.YC,mygrid.Depth);[X,Y,B]=convert2pcol(mygrid.XC,mygrid.YC,mygrid.Depth);
% figure; pcolor(X,Y,B); shading flat; colormap haxby; colorbar;xlim([-180,180]);
% Bb = B; Bb(B<500) = nan;
% figure; pcolor(X,Y,Bb); shading flat; colormap haxby; colorbar;xlim([-180,180]);
% figure; contourf(X,Y,B,[0,250,500,1000:1000:5000]); colormap haxby; colorbar;xlim([-180,180]);
% figure; histogram(B(:),0:20:500)
%%
B2=B;B2(B2==0)=nan;

% figure; pcolor(X,Y,B2); shading flat; colormap haxby; colorbar;xlim([-180,180]);
% figure; contourf(X,Y,B2); colormap haxby; colorbar;xlim([-180,180]);

%% Close off Med Sea, which is open in ECCO4r3 by just 1 point
[~,nx] = min(abs( (X(:)-(-5.5)).^2+(Y(:)-36).^2));
[nx,ny] = ind2sub(size(X),nx);
B2(nx,ny) = nan;
% figure; pcolor(X,Y,B2); shading flat; colormap haxby; colorbar;xlim([-180,180]);

%% Close off Southern Ocean east of Soouth Africa tip, at 21.5E
lon0 = 21.5; %[deg]E
for lat0=-33:-0.1:-71
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    B2(nx,ny) = nan;
end
% figure; pcolor(X,Y,B2); shading flat; colormap haxby; colorbar;xlim([-180,180]);

%% Close off Southern Ocean west of Drake Passage, at -66.5E
lon0 = -66.5; %[deg]E
for lat0=-55:-0.1:-67
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    B2(nx,ny) = nan;
end
% figure; pcolor(X,Y,B2); shading flat; colormap haxby; colorbar;xlim([-180,180]);

%% Close off Arctic Ocean north of 67.5N (where grid is Cartesian)
lat0 = 67.5; %[deg]E
for lon0=-190:0.01:190
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    B2(nx,ny) = nan;
end
% figure; pcolor(X,Y,B2); shading flat; colormap haxby; colorbar;xlim([-180,180]);

%% Land+Basin-boundaries Mask
Mask = isnan(B2)*2; %So==0 in ocean points, and ==2 on land points.
Mask(X<-180) = 2; Mask(X>180) = 2; %Since the convert2pcol creates a larger matrix in the longitude direction, i.e., -578 to ~603

%% Create Atlantic Mask

AtlSeedLon = -20; AtlSeedLat = 0; %[deg] E/N
[~,AtlSeedX] = min(abs( (X(:)-AtlSeedLon).^2+(Y(:)-AtlSeedLat).^2));
[AtlSeedX,AtlSeedY] = ind2sub(size(X),AtlSeedX);
MaskAtl = CalcFloodFill_V2(Mask,AtlSeedX,AtlSeedY);
fh = figure; contourf(X,Y,MaskAtl); colormap haxby; colorbar;xlim([-180,180]);
print(fh,[OutputFolder,'MaskAtl.png'],'-dpng','-r0'); 

%% Create Indo-Pacific Mask
%Western hemisphere flood-fill:
PacSeedLon = -150; PacSeedLat = 0; %[deg] E/N
[~,PacSeedX] = min(abs( (X(:)-PacSeedLon).^2+(Y(:)-PacSeedLat).^2));
[PacSeedX,PacSeedY] = ind2sub(size(X),PacSeedX);
MaskPac = CalcFloodFill_V2(Mask,PacSeedX,PacSeedY);
%Eastern hemisphere flood-fill:
PacSeedLon = 150; PacSeedLat = 0; %[deg] E/N
[~,PacSeedX] = min(abs( (X(:)-PacSeedLon).^2+(Y(:)-PacSeedLat).^2));
[PacSeedX,PacSeedY] = ind2sub(size(X),PacSeedX);
MaskPac = CalcFloodFill_V2(MaskPac,PacSeedX,PacSeedY);
fh = figure; contourf(X,Y,MaskPac); colormap haxby; colorbar;xlim([-180,180]);
print(fh,[OutputFolder,'MaskPac.png'],'-dpng','-r0'); 

%% Save masks
save([OutputFolder,'BasinMasks.mat'],'X','Y','MaskAtl','MaskPac');

%%

function fnout = fnos(fnin,os)
    if strcmp(os,'Linux')
       if or(strcmp(fnin(1:length('D:\Aviv\')),'D:\Aviv\'),strcmp(fnin(1:length('D:/Aviv/')),'D:/Aviv/'))
           fnout = ['/data/',fnin(length('D:\Aviv\')+1:end)];
       end
    else
        fnout = fnin;
        fnout = strrep(fnin,'\','/');
    end
end



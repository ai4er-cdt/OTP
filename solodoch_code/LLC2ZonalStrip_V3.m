%Aviv 2021-02-15; Pull out a zonal strip from ECCOV4r3 data. Works
%northward of -70N, and south of (?) 50N, i.e., where the gris is lon-lat.
%Aviv 2022-03-12; Correct the SizeMin-SizeMax allocations to lon-lat, i.e.,
%if #"lats">90, the previous (V2) allocation of #lats=SizeMin is wrong (as #lons=90)
function [LonNew,LatNew,VarNew] = LLC2ZonalStrip_V3(mygrid,Var,lat1,lat2)

nfaces = [1,2,4,5];
% lat1 = -70; lat2 = -50;
LatA = mygrid.YC{1};
[Ind1,Ind2] = find((LatA>=lat1).*(LatA<=lat2));
Size1 = max(Ind1) - min(Ind1) + 1;
Size2 = max(Ind2) - min(Ind2) + 1;
SizeMin = min(Size1,Size2);
SizeMax = max(Size1,Size2);
if SizeMin==90
    VarNew = nan([SizeMax,SizeMin*4]); %SizeMin==90 = longitude dim length
else
    VarNew = nan([SizeMin,SizeMax*4]); %SizeMax==90 = longitude dim length
end
[~,Ncols] = size(VarNew);
LonNew = VarNew; LatNew = VarNew; 
col1 = 1; col2 = Ncols/4;
for n=nfaces
    LonA = mygrid.XC{n}; LatA = mygrid.YC{n}; VarFace = Var{n}; 
    [Ind1,Ind2] = find((LatA>lat1).*(LatA<lat2));
    Ind1 = min(Ind1):max(Ind1); Ind2 = min(Ind2):max(Ind2);
    LonA = LonA(Ind1,Ind2); LatA = LatA(Ind1,Ind2); VarFace = VarFace(Ind1,Ind2); 
%     figure;contourf(LatA); title([num2str(n),', lat']); colorbar;
%     figure;contourf(LonA); title([num2str(n),', lon']); colorbar;
    if length(Ind1)==90; LonA = LonA'; LatA = LatA'; VarFace = VarFace'; end
    if LonA(1,1)>LonA(1,2); VarFace = fliplr(VarFace); LonA = fliplr(LonA); LatA = fliplr(LatA); end
    [a,~] = size(LatA); 
    if a>1; if LatA(1,1)>LatA(2,1); VarFace = flipud(VarFace); LonA = flipud(LonA); LatA = flipud(LatA); end; end
    VarNew(:,col1:col2) = VarFace; LonNew(:,col1:col2) = LonA; LatNew(:,col1:col2) = LatA;
    col1 = col2 + 1; col2 = col1 + Ncols/4 - 1;
end
% figure; contourf(VarNew); colorbar;

[~,IndSort] = sort(LonNew(1,:)); % Move longitude 180deg line to end of domain
LonNew = LonNew(:,IndSort); LatNew = LatNew(:,IndSort); VarNew = VarNew(:,IndSort);

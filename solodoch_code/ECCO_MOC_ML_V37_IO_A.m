%Aviv 2021-02-15
%V4, add option to use ECCOV4r4 (instead of r3) daily fields (only monthly %exist in r3 case)
%V5 2021-11, add option to do either bottom or intermediate MOC cell
%V6 2021-11-29: a. Correct bugs in V5 regarding cells; b. Add option to use multiple co-variates, e.g., SSH and BP
  %c. Add rng seed setting for debugging
%V7 2021-12-01. Adding option for AMOC (rathern than all lons 0-360 deg)
%V8 2021-12-06: Adding ability to do multiple layer NNs
%V9 2021-12-13: Make it operable on my Linux laptop
%V10 2021-12-14: Add a trainFcn parameter
%V11 2021-12-15: a. Corrected bug in LLC2ZonalStrip_V1.m->V2; b. Using basin masks (which I made in my MakeMasks_Va.m script today) to limit covariates to specific basins.
%V11 2021-12-19: a. Add option to include zonal wind as a zonal integral only (omit zonal dependence), i.e., as integrated Sverdrap Transport
%V12: a. Made a nonzero mse target. b. Added a Coefficient of Determination metric in plots. c. Added regularization option
%V13: a. Add option for longitudinal decimation, to decrease NN size
%V14: a. add option to disable NN training window popup; b. More flexible Decimation specification.
%V15: Allow Indo-Pacific sector
%V16: Allow delay-samples (poor-man's RNN).
%V16_IO: 2022-03-12 Allow arbitrary input variables for easy parameter sweeps with external-interface
% V17: 2022-03-15. (a) Add OutFoldParseByLat; (b) Save to mat file also the predictant (MOC, y) and NN predictions (output, yp) time series, so you don't have to relaod and
% preprocesses input data anytime you want to analyze the output yp vs predictant y.
%V18: 2022-03-22: (a) Allow loading SST and SSS; (b) Allow OutFoldParseByDelay in addition to OutFoldParseByLat & OutFoldParseByFiltTime
%V19: 2022-04-13. (a) Add dlat to the output folder name; (b) Add option for "Extended" Atlantic Ocean mask (i.e., including Med Sea), since the AMOC time series was calculated on the extended basin.
%V20: 2022-04-29. Chnage naming of basins. Basin==1 specifies MOC data and covariate masks corresponding to the Atlantic without Med.
%V21: 2022-05-14. (a) Name bug correctoin for trainbr cases with nmse_goal=0.
%V22: 2022-05-23. (a) Add Psi6 option: diagnose MOC by average value across a preseribed latitude range, on the density level giving max of the time-mean
% (b) Simplify output folder naming conventions for Psi's on a latitude range; (c) correct bug in delayed samples option (ddid not work with multiple input vars).
%V23: 2022-06-06. (a) Corrected bug: forgot to detrend scalar (zonal-mean) covariates. (b) For 3 or less zonal points, detrend3 didn't work, so I loop and use detrend
% (c) Corrected bug: covariates were not deseasoned (just detrended), while the MOC time series was. Adding switches to specify desired operations.
% (d) Removed SAM lines from code.
%V33: (a) Corrected errors with deseaoning and detrending of input X&Y. (b) Made output folder name dependent on choices for deseasnong and detrending.
%V34: output folder name dependent on training set size.
%V35: (a) control Levenberg-Marquardt Algorithm convergence speed via mu parameters. Already did so in previous version, but not for trainbr (which uses trainlm). 
% (b) Made mu params choices reflected in output filder name
%V36 2022-07-14: (a) Corrected bug - zonal smoothing was done only after collapsing multiple covariates' zonal structures to one dimension
% (b) Allowing non-consecutive time-delays
%V37 2022-08-08. (a) Fix flags for chossing between NNs and linear regression

%clear all
function ECCO_AABW_Regress_ZonStrip_V37_IO_A(varargin)
addpath(genpath('/data/Research/MATLAB_Scripts/ClimateDataToolbox/'));
DispFits = 0; %Show values of MSE, correlation , etc
addpath(genpath('/data/Research/MATLAB_Scripts/Utilities/'));

FigVis = 0; if FigVis==0; set(groot, 'DefaultFigureVisible', 'off'); end
TrainNN_Vis = 0; %If ==0 -> disable NN training window popup

OS = 'Linux';%'Windows';%
LoadPrev = 0; 
OutFoldParseByLat = 1; OutFoldParseByFiltTime = 0; OutFoldParseByDelay = 0;

LinReg = 0; LinRegGrad = 0; LinRegRobust = 0; %if LinReg==1, use linear regression, i.e., most of the following flags are not used
trainFcn = 'trainbr';%'trainbfg';%'traingdx';%;%'trainbr'; %'trainlm';%'trainscg';
if or( strcmp(trainFcn,'trainlm'), strcmp(trainFcn,'trainbr') )
    mu = [];%1; mu_dec = 0.5; mu_inc = 10; 
    % Put mu=[] to use default values for all 3 params.
    %Mu controlls convergence rate (higher is slower). default vals: initial mu=0.005; mu_dec(rease rate)=.1, mu_inc(rease rate)=10
end
if strcmp(trainFcn,'trainbr'); Reg = 0; else; Reg = 0.75; end %Network cost function regulariztion parameter (0-1). 0 corresponds to no regularization
% Reg = 0;
nmse_goal = 0;%.1;%0.1;%0.1;%0.1;%0.1;%.1;%0.1;%0.01; % %Normalized MSE Performance Goal. A finite value can help the training process converge.
ActivFunc = 'poslin';%'purelin';%'tansig';%(poslin is relu; tansig is tanh)
NetNorm = 'mapstd';%'minmax';%'Global';
divideFcn = 'divideind';%'dividerand';%'divideint';% %in divideind I pick the division indices; divideint assigns indices in an interleaved fashion
TrainValidTestFracs = [0.7,0.3];%[0.4,0.3,0.3];%
rng_seed = 1; %set to -1 for no seed setting. 

NNNlayers = 1; %# NN hidden layers
NneuronsList = [1,3,5];%[1,5,10,15,20];%[1,3,5];%[1,5,10,15,20];%[20];%%[5,10,15,20,25,30];%[3,5,8,10,15,20];%10;%[5,10,15,20,30];%[40,50,60];%,50,100];%[4,6,8,10,15,20];%[6,8:4:20];% 6:20;%10;%2:10;%[1,2:2:6];%ones([1,4]);%[3,3,3];%[8,8,8,8,8,8];
NNrepeats = 5; if LinReg==1; NneuronsList = []; end

ECCOv = 'V4r3';%'V4r4';%
EccoYear0 = 1992;
BotttomMOC = 1; %==1 (0) for bottom (top) MOC cell
Basin = 2; %==1 for AMOC/Atl-basin, 1.5 for Atl covariates and Atl-Med AMOC; 1.75 for Atl+Med AMOC+covariates; ==2 for PMOC/Indo-Pacific, ==0 for all lons (0-360deg)
PsiMethod = 4;%1 %Definition of MOC time series (i.e., how to get a single MOC strength per time sample from the MOC(lat,density) distribution
CoVariateNames = {'OBP'};%,'ETAN','oceTAUX_int'};%'SIheff';%'THETA_S','SALT_S'};%{'OBP','ETAN','oceTAUX_int','THETA_S','SALT_S'};%};%{'OBP','oceTAUX_int'};%{'ETAN','oceTAUX_int'};% {'OBP'};%'OBP'};%{'oceTAUX_int'};%"_int" means zonal integral%{'OBP','ETAN','oceTAUX'};%'OBP'};%,'ETAN'};%'oceTAUX';%'DensFlux';%'oceTAUY';%'PHIBOT';%%%'OBPNOPAB';%'ETAN';%  'PHIBOT';% 
DetrendCovar = 1; DeseasonCovar = 1; DetrendMOC = 2; DeseasonMOC = 2;
DelaySamples = 0;%[3,6,9,12];%0; %==0: only instantenous samples; >=1->==number of additional delays; <=-1 -> (-#) of single-sample delay
FiltType = '';%'';%'LPF';%
LPF_Freq = 1/18;%1/6;%1/(2*12); %[1/month]
ZonalSmoothRadDeg = 0;%2;%0; %[deg lon]
LonDec = 1;%-20;%10; %Decimate (subsample) #zonal points of each covariate by this factor. i.e., for LonDec==1 no decimation is done.
BathThresh = 0;%;%150; %[m] Depth threshold for grid cell usability

dens_sep = 1036.99;%[kg/m^3] isopycnal separating upper and lower MOC branches 
if BotttomMOC==1 %Deep MOC cell
    latrange = [-60 -40]; lat0 = -60;% -50; %For MOC time series
    if Basin==0; DenseVal = 1037.1; elseif Basin==2; DenseVal = 1037.05; else; error('Unspecified density for this basin'); end
    dlat = 2;%lat1 = -62; lat2 = -58; %For Covariate
%     lat1 = -22; lat2 = -18; %For Covariate
else %Intermediate MOC cell
    if Basin==1; DenseVal = 1036.8; else; error('Unspecified density for this basin'); end
    lat0 = 26; %latrange = [21,31]; %For MOC time series
    dlat = 1;% lat1 = lat0-dlat; lat2 = lat0+dlat; %For Covariate
    latrange = [lat0 - dlat,lat0 + dlat];
end

%% Use variable-length input variable list
%Input expected in dictionary-like format, e.g., 
 %  ECCO_AABW_Regress_ZonStrip_V16_IO({'PsiMethod',4},{'LonDec',2},{'FiltType','LPF'});
for nv=1:nargin
%     disp(['eval(varargin{',num2str(nv),'}{1}=varargin{',num2str(nv),'}{2});']);
%     disp(['eval(varargin{',num2str(nv),'}{1}=varargin{',num2str(nv),'}{2};)']);
    eval([eval(['varargin{',num2str(nv),'}{1}']),'=varargin{',num2str(nv),'}{2}']);
end
lat1 = lat0-dlat; lat2 = lat0+dlat; %For Covariate

if LPF_Freq==1; FiltType = ''; end % A hack to make external parameter sweeps on LPF freq easier to handle
%%

if Basin>0
    LonDecPeriodic = 0; %If (==1) and applying decimation, treat first and last zonal points as ~identical.
else
    LonDecPeriodic = 1; %If (==1) and applying decimation, treat first and last zonal points as ~identical.
end
% PsiMethod = 5; DenseVal = 1037.1; latrange = [-60 -50]; lat1 = latrange(1); lat2 = latrange(2); %For Covariate
CorrType = 'Corr';%'PartialCorr';%'LagCorr'; 
LoadPrevNN = 0;
FnPrevNN = fnos('D:/Aviv/Research/AABWvsACC/ECCO/MocRegression/ECCOV4r3/PHIBOTVsMocPsi4_LPF6_ZonSmoothSig2_NN10_rep1.mat',OS);
% FnPrevNN = 'D:/Aviv/Research/AABWvsACC/ECCO/MocCorrelates/PHIBOTVsMocPsi4_LPF6__NN10_rep1.mat';

if strcmp(ECCOv,'V4r4')
    Year1 = 1992; Year2 = 2016; Month1 = 1; Month2 = 12;
    Years = Year1:Year2; Months = Month1:Month2;
    YearsVec = reshape(repmat(Years,[length(Months),1]),[1,length(Years)*length(Months)]);
    MonthsVec = reshape(repmat(Months,[1,length(Years)]),[1,length(Years)*length(Months)]);
    Nsamps = (Year2-Year1+1)*(Month2-Month1+1);
    startdate = datenum('1992-01-01'); enddate = datenum('2016-12-31'); 
else
    Nsamps = 288;
end
if strcmp(ECCOv,'V4r4')
    addpath ./ECCOv4r4Andrew; 
    dirv4r4 = fnos('D:/Research/NumericalModels/ECCO/Version4/Release4/',OS); 
    if Basin==
        if DetrendMOC*DeseasonMOC>=1
            FN_MOC_Inds = fnos('D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/DeepSoMOCtot_deseas_detrend_Inds2d_ECCOV4r4.mat',OS);
        else
            error('Where is Var4r4 raw MOC?');
        end
    else
        error('V4r4 AMOC/PMOC Files do not exist yet');
    end
else
    dirv4r3 = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/',OS); 
    deseadetrens_str = ''; if DeseasonMOC>=1; deseadetrens_str = '_deseas'; end
    if DetrendMOC>=1;  deseadetrens_str = [deseadetrens_str,'_detrend']; end
    if Basin==0
        FN_MOC_Inds = fnos(['D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/DeepSoMOC',deseadetrens_str,'_Inds.mat'],OS);
    else
        if floor(Basin)==1.5
            FN_MOC_Inds = fnos(['D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/AtlAndMed/AMOCMed',deseadetrens_str,'_Inds.mat'],OS);
        elseif floor(Basin)==1
            FN_MOC_Inds = fnos(['D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/Atl/AMOC',deseadetrens_str,'_Inds.mat'],OS);
        elseif Basin==2
            FN_MOC_Inds = fnos(['D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/Pac/IndoPac',deseadetrens_str,'_Inds.mat'],OS);
        else
            error('No such basin option');
        end
    end
end
p = genpath(fnos('D:/Aviv/Research/MATLAB_Scripts/Ocean/gcmfaces/',OS)); addpath(p);
dirGrid = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/nctiles_grid/',OS);
if Basin==1
    BasinMasksFN = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/BasinMasks.mat',OS);
else
    BasinMasksFN = fnos('D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/BasinMasksWithMed.mat',OS);
end
FN_DensFlux = fnos('D:/Aviv/Research/AABWvsACC/ECCO/MocCorrelates/ECCO4r3_SurfDensFlux.mat',OS);

if or(DetrendCovar,DeseasonCovar); deseadetrens_str = '_Xd'; 
    if DetrendCovar==1; deseadetrens_str = [deseadetrens_str,'T']; end
    if DeseasonCovar==1; deseadetrens_str = [deseadetrens_str,'S']; end
end
if or(DetrendMOC,DeseasonMOC); deseadetrens_str = [deseadetrens_str,'_Yd']; 
    if DetrendMOC>=1; deseadetrens_str = [deseadetrens_str,'T']; end; if DetrendMOC>1; deseadetrens_str = [deseadetrens_str,'2']; end
    if DeseasonMOC>=1; deseadetrens_str = [deseadetrens_str,'S']; end; if DeseasonMOC>1; deseadetrens_str = [deseadetrens_str,'2']; end
end
OutputFolder = fnos(['D:/Aviv/Research/AABWvsACC/ECCO/MocRegression/ECCO',ECCOv,filesep,'LatSpecific',deseadetrens_str,filesep,'Reps',num2str(NNrepeats),filesep],OS);

if BotttomMOC==1
    BotttomMOCstr='Deepcell_'; BotttomMOCstrTitle='Deep cell '; 
else
    BotttomMOCstr = 'Topcell_'; BotttomMOCstrTitle = 'Top cell '; 
end
if Basin==1
    BasinStr='Atl_'; BasinStrTitle='Atl '; 
elseif Basin==1.5
    BasinStr='AtlMed_'; BasinStrTitle='Atl and Med '; 
elseif Basin==1.75
    BasinStr='AtlMed2_'; BasinStrTitle='Atl and Med x2 '; 
elseif Basin==2
    BasinStr='IndoPac_'; BasinStrTitle='IndoPac '; 
else
    BasinStr=''; BasinStrTitle=''; 
end

%% "global" loads
nFaces = 5; %nFaces is the number of faces in this gcm set-up of current interest.
fileFormat = 'nctiles';%'compact'; %fileFormat is the file format ('straight','cube','compact')
memoryLimit=0; omitNativeGrid=isempty(dir([dirGrid 'tile001.mitgrid']));
grid_load(dirGrid,nFaces,fileFormat,memoryLimit,omitNativeGrid); %Load the grid.
gcmfaces_global; % Define global variables


%%
if ZonalSmoothRadDeg>0
    TitleFiltLon = [', ',num2str(ZonalSmoothRadDeg),'deg zonal smooth rad']; FnFiltLon = ['_ZonSmoothSig',num2str(ZonalSmoothRadDeg)]; 
else
    TitleFiltLon = ''; FnFiltLon = '';
end 
switch CorrType
            case 'Corr'
            TitleCorrDesc = ''; FnCorrDesc = '_';
            case 'LagCorr'
            TitleCorrDesc = 'Lagged-'; FnCorrDesc = '_Lagged';
            case 'PartialCorr'
            TitleCorrDesc = 'Partial-'; FnCorrDesc = '_Part';
end
switch FiltType
    case ''; TitleFiltTime = ''; FnFiltTime = ''; FiltWind = 0;
    case 'LPF'; TitleFiltTime = [', ',num2str(1/LPF_Freq),'months LPF']; FnFiltTime = ['_LPF',num2str(1/LPF_Freq)]; FiltWind = LPF_Freq;
end 
PsiStr = ['Psi',num2str(PsiMethod)];
if PsiMethod==5
    PsiStr = [PsiStr,'Dens',num2str(DenseVal)];
end
LatStr = ['lat',num2str(lat0)];

NCovars = length(CoVariateNames); 
% CoVariateNamesStr = CoVariateNames{1}; for nc=2:NCovars; CoVariateNamesStr = [CoVariateNamesStr,CoVariateNames{nc}]; end
CoVariateNamesStr = CoVariateNames{1}; for nc=2:NCovars; CoVariateNamesStr = [CoVariateNamesStr,'+',CoVariateNames{nc}]; end
% CoVariateNamesStrTitle = CoVariateNames{1}; for nc=2:NCovars; CoVariateNamesStrTitle = [CoVariateNamesStrTitle,'+',CoVariateNames{nc}]; end
if rng_seed>=0; seedn = rng_seed; else; seedn = rng; seedn = seedn.Seed; end; seedn = num2str(seedn);
RegStr = ''; if Reg>0; RegStr = ['_Reg',num2str(Reg)]; end
NmseStr = ''; if nmse_goal>0; NmseStr = ['_NMSE',num2str(nmse_goal)]; end
LonDecStr = ''; 
if LonDec~=1
    if LonDec==0; error('LonDec=0'); end
    LonDecStr = ['_LonDec',num2str(LonDec)]; 
    if exist('DecimStartInd','Var'); LonDecStr = [LonDecStr,'Start',num2str(DecimStartInd)]; end
end    
DelayStr = ''; if sum(DelaySamples~=0)>0; DelayStr = ['_Delay',num2str(DelaySamples(1))]; for nd=2:length(DelaySamples); DelayStr = [DelayStr,'-',num2str(DelaySamples(nd))]; end; end
dlatstr = ['_dlat',num2str(dlat)];
IndsStr = ''; 
for n=1:length(TrainValidTestFracs)
    IndsStr = [IndsStr,num2str(TrainValidTestFracs(n))];
end
IndsStr = strrep(IndsStr,'.','');
OutFoldSec = '';
if OutFoldParseByLat==0
    PsiStr = [PsiStr,LatStr];
else
    OutFoldSec = LatStr;
end
trainFcnStr = trainFcn;
if or( strcmp(trainFcn,'trainlm'), strcmp(trainFcn,'trainbr') )
    if ~isempty(mu); trainFcnStr = [trainFcnStr,'-mu',num2str(mu),'-',num2str(mu_dec),'-',num2str(mu_inc)]; end
end
if LinReg==1
    RegType = ['LinReg_',divideFcn,IndsStr,LonDecStr,dlatstr];
else
    RegType = ['NN_',num2str(NNNlayers),'layers_',ActivFunc,'_',trainFcnStr,RegStr,NmseStr,'_',NetNorm,'_',divideFcn,IndsStr,LonDecStr,dlatstr];
end
if OutFoldParseByFiltTime==1
    OutFoldSec = [OutFoldSec,FnFiltTime];
    OutputFolder = [OutputFolder,BasinStr,BotttomMOCstr,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltLon,FnCorrDesc,RegType];
else
    OutputFolder = [OutputFolder,BasinStr,BotttomMOCstr,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,RegType];
end
if BathThresh>0; OutputFolder = [OutputFolder,'_Bath',num2str(BathThresh)]; end
if OutFoldParseByDelay==0
    OutputFolder = [OutputFolder,DelayStr,'_seed',seedn,filesep];
else
    OutputFolder = [OutputFolder,'_seed',seedn,filesep];
    OutFoldSec = [OutFoldSec,DelayStr];
end

if ~isempty(OutFoldSec)
    OutputFolder = [OutputFolder,filesep,OutFoldSec,filesep];
end

if ~exist(OutputFolder,'dir')
    mkdir(OutputFolder);
end
disp(['OutputFolder=']); disp(OutputFolder);

%% Generate Lon-Lat grid
% fileName= [dirv4r3 'nctiles_monthly/PHIBOT/' 'PHIBOT']; fldName='PHIBOT'; MONTH = 1;
% PHIBOT=read_nctiles(fileName,fldName,MONTH,1); %need to plug in 1 for surface depth if its a 4d field
% [Lon,Lat,~] = LLC2ZonalStrip_V3(mygrid,PHIBOT,lat1,lat2);
[Lon,Lat,Bath] = LLC2ZonalStrip_V3(mygrid,mygrid.Depth,lat1,lat2);
[Nlat,Nlon] = size(Lon);
lon = Lon(1,:);
bath = mean(Bath,1);
if or(BathThresh>0, Basin>0)
    %==1 for AMOC/Atl-basin, ==1.5 for Atl+Med Sea, ==2 for PMOC/Indo-Pacific, ==0 for all lons (0-360deg)
    CoVariateMask = ones(Nlat,Nlon); 
    if floor(Basin)==1
        load(BasinMasksFN,'X','Y','MaskAtl'); MaskBasin = MaskAtl;
    elseif Basin==2
        load(BasinMasksFN,'X','Y','MaskPac'); MaskBasin = MaskPac;
    elseif Basin~=0
        error('');
    end
    if Basin>0
        for nlat=1:Nlat
            lat = unique(Lat(nlat,:));
            [nx,ny]=find(Y==lat);
            if length(unique(ny))~=1
                error('You are in a non-cartesian part of grid');
            else
                ny = unique(ny);
                nx1 = find(X(:,ny)==min(lon));
                nx2 = find(X(:,ny)==max(lon));
                CoVariateMask(nlat,:) = MaskBasin(nx1:nx2,ny);
            end
        end
    end
    if BathThresh>0
        CoVariateMask(Bath<BathThresh) = 0;
    end
    CoVariateMask(CoVariateMask~=1) = 0; %Some boundaries are marked by 2 rather than 0, so mark them in 0 as well.
    [~,nx]=find(CoVariateMask==1);
    mask_nx1 = min(nx); mask_nx2 = max(nx); mask_nx = unique(nx);%mask_nx = mask_nx1:mask_nx2;
    CoVariateMask = CoVariateMask(:,mask_nx);
    CoVariateMask(CoVariateMask==0) = nan; %Put nan's on mask
    Lon = Lon(:,mask_nx); Lat = Lat(:,mask_nx);
    [Nlat,Nlon] = size(Lon);
    lon = lon(mask_nx); bath = bath(mask_nx);
end

%%
DoZonalIntegral = zeros([1,NCovars]);
for nc=1:NCovars
   covname = CoVariateNames{nc}; 
   if length(covname)>4
       if strcmp(covname(end-3:end),'_int')
           DoZonalIntegral(nc) = 1; CoVariateNames{nc} = covname(1:end-4);
       end
   end
end
NumInts = sum(DoZonalIntegral);
NCovarsVec = NCovars-NumInts;
CoVariate = nan(Nlat,Nlon,NCovarsVec,Nsamps); 
CoVariateScalars = nan(NumInts,Nsamps); 

if sum(strcmp(CoVariateNames,'DensFlux'))
    nc = find(strcmp(CoVariateNames,'DensFlux'));
    if LoadPrev==1 %isfile(FN_DensFlux) 
        load(FN_DensFlux,'alldensflux');
        CoVariate(:,:,nc,:) = alldensflux; clear alldensflux;
        A=read_nctiles([dirv4r3 'nctiles_monthly/PHIBOT/PHIBOT'],'PHIBOT',1,1); %need to plug in 1 for surface depth if its a 4d field
        [X,Y,~]=convert2pcol(mygrid.XC,mygrid.YC,A); clear A;
    end
end

%% Reading 2D NC fields (ETAN is the liquid sea surface height)
if LoadPrev==0
    for nt=1:Nsamps
        if strcmp(ECCOv,'V4r4')
            year = num2str(YearsVec(nt)); month = num2str(MonthsVec(nt));
%             thedate = startdate + nt - 1; fileidstr = ['_',datestr(thedate,'yyyy_mm_dd')];    
        end
        nc_vecs = 0; nc_scals = 0;
        for nc=1:NCovars
            CoVariateName = CoVariateNames{nc};
            switch CoVariateName
                case 'oceTAUX'
                fldName='oceTAUX';
                fileName= [dirv4r3 'nctiles_monthly/oceTAUX/' 'oceTAUX'];
                fldx=read_nctiles(fileName,fldName,nt); %Read in the nt-th monthly record of ETAN
                fldName='oceTAUY'; 
                fileName= [dirv4r3 'nctiles_monthly/oceTAUY/' 'oceTAUY'];
                fldy=read_nctiles(fileName,fldName,nt); %Read in the nt-th monthly record of ETAN
                [fldUe,fldVn]=calc_UEVNfromUXVY(fldx,fldy);
                [~,~,CoVariate_t] = LLC2ZonalStrip_V3(mygrid,fldUe,lat1,lat2);
                case 'oceTAUY'
                fldName='oceTAUX';
                fileName= [dirv4r3 'nctiles_monthly/oceTAUX/' 'oceTAUX'];
                fldx=read_nctiles(fileName,fldName,nt); %Read in the nt-th monthly record of ETAN
                fldName='oceTAUY'; 
                fileName= [dirv4r3 'nctiles_monthly/oceTAUY/' 'oceTAUY'];
                fldy=read_nctiles(fileName,fldName,nt); %Read in the nt-th monthly record of ETAN
                [fldUe,fldVn]=calc_UEVNfromUXVY(fldx,fldy);
                [~,~,CoVariate_t] = LLC2ZonalStrip_V3(mygrid,fldVn,lat1,lat2);
                case 'DensFlux'
                fileName= [dirv4r3 'nctiles_monthly/SALT_S/' 'SALT_S']; fldName='SALT';
                fld=read_nctiles(fileName,fldName,nt,1); %need to plug in 1 for surface depth if its a 4d field
                [X,Y,SALT]=convert2pcol(mygrid.XC,mygrid.YC,fld); 

                fileName= [dirv4r3 'nctiles_monthly/THETA_S/' 'THETA_S']; fldName='THETA';
                fld=read_nctiles(fileName,fldName,nt,1); %Read in the nt-th monthly record of ETAN
                [X,Y,THETA]=convert2pcol(mygrid.XC,mygrid.YC,fld); 

                fileName= [dirv4r3 'nctiles_monthly/SFLUX/' 'SFLUX']; fldName='SFLUX';
                fld=read_nctiles(fileName,fldName,nt); %need to plug in 1 for surface depth if its a 4d field
                [X,Y,SFLUX]=convert2pcol(mygrid.XC,mygrid.YC,fld); 

                fileName= [dirv4r3 'nctiles_monthly/TFLUX/' 'TFLUX'];fldName='TFLUX';
                fld=read_nctiles(fileName,fldName,nt); %need to plug in 1 for surface depth if its a 4d field
                [X,Y,TFLUX]=convert2pcol(mygrid.XC,mygrid.YC,fld); 

                Press_db = 10*ones(size(SALT)); %[db]
                % dens=densmdjwf(SALT,THETA,Press_db);
                dens = sw_dens(SALT,THETA,Press_db);
                Cp=sw_cp(SALT,THETA,Press_db);

                % [alpha,beta]=calcAlphaBeta(SALT,THETA,5*ones(1080,360));
                [alpha] = sw_alpha(SALT,THETA,Press_db, 'ptmp');
                [beta] = sw_beta(SALT,THETA,Press_db, 'ptmp');
                CoVariate_t = beta.*SFLUX-alpha.*TFLUX./Cp; %change this to sflux and tflux
                otherwise 
                    if strcmp(ECCOv,'V4r3')
                        fileName= [dirv4r3 'nctiles_monthly/',CoVariateName,'/',CoVariateName];
                    else
                        fileName= [dirv4r4 'nctiles_monthly/',CoVariateName,'/',CoVariateName,'_',year,'_',month,'.nc'];
                    end
                    if strcmp(CoVariateName(end-1:end),'_S'); fldName = CoVariateName(1:end-2); else; fldName = CoVariateName; end
                    CoVariate_t = read_nctiles_V4r3r4(fileName,fldName,nt,1,ECCOv);
                    [~,~,CoVariate_t] = LLC2ZonalStrip_V3(mygrid,CoVariate_t,lat1,lat2);
            end
            if exist('CoVariateMask','var')
                CoVariate_t = CoVariate_t(:,mask_nx).*CoVariateMask;
            end
            if DoZonalIntegral(nc)==0 %So this variable includes the full zonal dependence
                nc_vecs = nc_vecs + 1;
                CoVariate(:,:,nc_vecs,nt) = CoVariate_t;
            else  %So this variable is to become a zonal-mean/sum
                nc_scals = nc_scals + 1;
                if strcmp(CoVariateName,'oceTAUX')
                    f = 2*pi*2/24/3600*sind(lat0); %[rad/s]
                    rho0 = 1025; % [kg/m^3]
                    Re = 6.378e6;% [m] Earth radius
                    CoVariateScalars(nc_scals,nt) = -squeeze(nansum(CoVariate_t,[1,2]))*pi/180*Re*cosd(lat0)/f/rho0; %[Sv]
                else
                    CoVariateScalars(nc_scals,nt) = squeeze(nansum(CoVariate_t,[1,2])); %whatever the units of the input variable
                end
            end
        end
    end
    if sum(strcmp(CoVariateNames,'DensFlux'))
        nc = find(strcmp(CoVariateNames,'DensFlux'));
        alldensflux = squeeze(CoVariate(:,:,nc,:)); save(FN_DensFlux,'alldensflux');  clear alldensflux; 
    end
end

%% MOC

if strcmp(ECCOv,'V4r4')
    warning('(Aviv) Warning, in V5 I have not taken care of upper cell/lower cell option in EccoV4r4');
    if PsiMethod==1; load(FN_MOC_Inds,'Psi1'); Psi_d = Psi1; clear Psi1;
    elseif PsiMethod==2; load(FN_MOC_Inds,'Psi2'); Psi_d = Psi2; clear Psi2;
    elseif PsiMethod==3; load(FN_MOC_Inds,'Psi3'); Psi_d = Psi3; clear Psi3;
    elseif PsiMethod==4; load(FN_MOC_Inds,'Psi4'); Psi_d = Psi4; clear Psi4;
    end
    Ndays = length(Psi_d);
    caldays = startdate + (0:Ndays-1);
    y1 = datestr(caldays(1)); y1 = str2double(y1(end-3:end));
    y2 = datestr(caldays(end)); y2 = str2double(y2(end-3:end));
    Nyears = y2 - y1 + 1;
    
    if 0
        caldays2 = datetime(datestr(caldays));
        [Psi_m, yr] = reshapetimeseries(caldays2, Psi_d', 'bin', 'month');
        Psi = reshape(Psi_m,[1,12*Nyears]); 
    else
        Psi = zeros([1,12*Nyears]); 
        PsiCum = Psi_d(1); mon0 = datestr(caldays(1)); mon0 = mon0(4:6); nmon = 1; dayspermonth = 1; 
        for nd=2:Ndays
            mon1 = datestr(caldays(nd)); mon1 = mon1(4:6);
            if nd==Ndays
                dayspermonth = dayspermonth + 1;
                Psi(nmon) = PsiCum/dayspermonth;
            elseif strcmp(mon0,mon1)
                dayspermonth = dayspermonth + 1;
                PsiCum = PsiCum + Psi_d(nd); 
            else
                Psi(nmon) = PsiCum/dayspermonth;
                nmon = nmon + 1; dayspermonth = 1; PsiCum = Psi_d(nd); 
            end
            mon0 = mon1;
        end
    end
    Nsamps = length(Psi);
else
    load(FN_MOC_Inds,'lat','PSI_notrend','dens_bnds');%,'pc1'
    latinds = find(lat==latrange(1)):find(lat==latrange(end)) ;
    nlat0 = find(lat==lat0);
    if BotttomMOC==1
        CellDensInds = find(dens_bnds>dens_sep); 
        PSI_notrend = -PSI_notrend; %make tranport values positive
    else
        CellDensInds = find(dens_bnds<dens_sep);
    end
    [~,~,Nsamps] = size(PSI_notrend);
    if PsiMethod==4
        PSI_Lat0_Mean = mean(PSI_notrend(nlat0,CellDensInds,:),3); 
        DensIndMax = find(PSI_Lat0_Mean==max(PSI_Lat0_Mean));  %Pick lat, then Pick density where time-mean is minimal %this could obviously have been vectorized, but for readbility, keeping in same form as options 1-3
    elseif PsiMethod==5
        [~,DensIndChoice] = min(abs(dens_bnds-DenseVal));  %Mean over lat range, at a particular density value choice
    elseif PsiMethod==6
        P = mean(PSI_notrend(latinds,CellDensInds,:),[1,3]); %Mean over time&lat range
        DensIndMaxLatRange = find(P==max(P)); clear P; %Find density where lat&time-mean is minimal
    end
    
    Psi = zeros([1,Nsamps]);%-10.6146
    for nt=1:Nsamps
        if PsiMethod==1
            Psi(nt)=max(squeeze(PSI_notrend(nlat0,CellDensInds,nt))); %Max over density at a given lat
        elseif PsiMethod==2
            Psi(nt)=max(max(PSI_notrend(latinds,CellDensInds,nt),[],2)); %Max over density and lat at a given lat range
        elseif PsiMethod==3
            Psi(nt)=min(max(PSI_notrend(latinds,CellDensInds,nt),[],2)); %Max over density, min over lat, at a given lat
        elseif PsiMethod==4
            Psi(nt)=PSI_notrend(nlat0,CellDensInds(DensIndMax),nt); %Pick lat, then Pick density where time-mean is minimal %this could obviously have been vectorized, but for readbility, keeping in same form as options 1-3
        elseif PsiMethod==5
            Psi(nt)=mean(PSI_notrend(latinds,DensIndChoice,nt),1); %Mean over lat range, at a particular density value choice
        elseif PsiMethod==6
            Psi(nt)=mean(PSI_notrend(latinds,CellDensInds(DensIndMaxLatRange),nt),1); %Mean over lat range, at density where lat&time-mean is minimal
        end
    end
end

%% Average in latitude. Remove nans
%%%Original def: CoVariate ~ (Nlat,Nlon,NCovars,Nsamps); 
CoVariateFlat = (mean(CoVariate,1)); %Don't squeeze, in case NCovars==1 (would like to squeeze just latitude). Rather, do: 
CoVariateFlat = reshape(CoVariateFlat,[Nlon,NCovarsVec,Nsamps]);%~ (Nlon,NCovars,Nsamps); 
x0 = sum(sum(CoVariateFlat,3),2);
IndsNotNanX = find(~isnan(x0)); CoVariateFlat = CoVariateFlat(IndsNotNanX,:,:); 
LonsNotNan = lon(IndsNotNanX); Nlon2 = length(LonsNotNan);

%% Zonal smoothing, if there are covariates which were not zonally averaged
if and(ZonalSmoothRadDeg>0,numel(CoVariateFlat)>0)
    WinL = 11;
    alpha = ((WinL-1)/2)/ZonalSmoothRadDeg;
    GaussFilter = gausswin(WinL,alpha); %figure;plot(w)
    GaussFilter = GaussFilter/sum(GaussFilter);
%     x = filtfilt(GaussFilter,1,x'); x = x';
    for nt=1:Nsamps
        for nv = 1:NCovarsVec
            CoVariateFlat(:,nv,nt) = filtfilt(GaussFilter,1,CoVariateFlat(:,nv,nt));% x = x';
        end
    end
end 

%% Decimate longitude
if LonDec>1
    if LonDecPeriodic==0
        nn = 1:LonDec:Nlon2; 
    else
        nn = 1:LonDec:(Nlon2-LonDec); %Periodic strip -> throw away last point since it is identical to first
    end
    LonsNotNan = LonsNotNan(nn); Nlon2 = length(LonsNotNan);
    CoVariateFlat = CoVariateFlat(nn,:,:); 
elseif LonDec<0 %In this case (-)LonDec is the number of points to keep, rather than the decimation ratio
    if LonDecPeriodic==0
        nn = round(linspace(1,Nlon2,(-LonDec)));
    else
%         nn = round(linspace(1,Nlon2,(-LonDec+1))); nn = nn(1:end-1); %Periodic strip -> throw away last point since it is identical to first
        nn = round(linspace(1,Nlon2,(-LonDec+1))); 
        if exist('DecimStartInd','Var')
            nn = mod(nn + DecimStartInd,Nlon2); nn(nn==0) = Nlon2;
        end        
        nn = nn(1:end-1); %Periodic strip -> throw away last point since it is identical to first
    end
    LonsNotNan = LonsNotNan(nn); Nlon2 = length(LonsNotNan);
    CoVariateFlat = CoVariateFlat(nn,:,:); 
end

%% Detrending
Y = zeros([1,Nsamps]); M = Y; D = Y;
for nt=1:Nsamps
    Y(nt) = EccoYear0 + fix((nt-1)./12); M(nt) = nt - (Y(nt)-EccoYear0)*12; D(nt) = eomday(Y(nt),M(nt))/2;
end
t = datetime(Y,M,round(D));

%If DetrendMOC and/or DeseasonMOC>=1, I am already loading PSI(lat,dens,time) which is
%detrended and/and deseasoned. If==2, repeat the operation after the MOC univariate time series definition
if DetrendMOC==2; Psi = detrend(Psi); end
if DeseasonMOC==2; Psi = deseason(Psi,t,'monthly'); end

%CoVariate = nan(Nlat,Nlon2,NCovarsVec,Nsamps); 
if numel(CoVariateFlat)>0 %For all covariates which were not zonally averaged
%         CoVariateFlat = detrend3(CoVariateFlat); 
    for nlon=1:Nlon2
        for nv = 1:NCovarsVec
            if DetrendCovar==1; CoVariateFlat(nlon,nv,:) = detrend(squeeze(CoVariateFlat(nlon,nv,:))); end
            if DeseasonCovar==1; CoVariateFlat(nlon,nv,:) = deseason(squeeze(CoVariateFlat(nlon,nv,:)),t,'monthly'); end
        end
    end
end
for nv = 1:NumInts %For all covariates which were zonally averaged
    if DetrendCovar==1; CoVariateScalars(nv,:) = detrend(squeeze(CoVariateScalars(nv,:))); end
    if DeseasonCovar==1; CoVariateScalars(nv,:) = deseason(squeeze(CoVariateScalars(nv,:)),t,'monthly'); end
end

%% Prepare data for regression
%% Normalize data for regression
if NetNorm=='Global'
    for nc=1:NCovarsVec
        CoVariateFlat_0 = squeeze(CoVariateFlat(:,nc,:)); %CoVariateFlat~(Nlon,NCovarsVec,Nsamps); 
        m = min(CoVariateFlat_0(:)); M = max(CoVariateFlat_0(:));
        CoVariateFlat_0 = (CoVariateFlat_0-(M+m)/2) / ((M-m)/2);
        CoVariateFlat(:,nc,:) = CoVariateFlat_0;
    end
    clear CoVariateFlat_0;
end
%% Reshape as [Nlon2*NCovarsVec,Nsamps]
CoVariateFlat = reshape(CoVariateFlat,[Nlon2*NCovarsVec,Nsamps]); %Reshape as [Nlon2*NCovarsVec,Nsamps]
CoVariateFlat = CoVariateFlat'; % Reshape as (Nsamps,Nlon2*NCovarsVec); 
if ~isempty(CoVariateScalars)
    CoVariateFlat(:,end+1:end+NumInts) = CoVariateScalars; %Add the zonal-mean variables, if defined.
end
x = CoVariateFlat; %Rename, since x may undergo filtering. Keeping CoVariateFlat allows to check regression against unfiltered data later
y = Psi'; 
Nlon_pseudo = Nlon2*NCovarsVec + NumInts;

%% Temporal Filtering/Smoothing
switch FiltType
    case 'LPF'; x = lowpass(x,LPF_Freq); y = lowpass(y,LPF_Freq);
end 

% nt = 10;
% figure; yyaxis left; plot(lon,x0(nt,:)); hold on; plot(lon,x(nt,:));ylabel('Bottom pressure');
% yyaxis right; plot(lon,-bath); ylabel('Depth [m]'); xlabel('Lon [deg]'); legend('raw','smooth','bath'); 

%% Poor-man's RNN, i.e., introduce delayed samples
% Do this both for the filtered (x,y) and unfiltered (asd,Psi) predictors and predictants
if sum(DelaySamples~=0)>1
    if all(DelaySamples>0)
        DelaySamplesMax = max(DelaySamples); DelaySamplesN = length(DelaySamples);
        %Prep predictors with less time samples (decreased by the number of delays), and
        %more "spatial" samples (multiply by number of delays)
        x_d = nan*zeros([Nsamps-DelaySamplesMax,Nlon_pseudo*(1+DelaySamplesN)]); CoVariateFlat_d = x_d;
        x_d(:,1:Nlon_pseudo) = x(1+DelaySamplesMax:end,:);
        %Populate first parts of padded arrays with undelayed samples
        CoVariateFlat_d(:,1:Nlon_pseudo) = CoVariateFlat(1+DelaySamplesMax:end,:);
        y_d = y(1+DelaySamplesMax:end); Psi_d = Psi(1+DelaySamplesMax:end);
        %Populate latter parts of padded arrays with delayed samples
        for nd=1:DelaySamplesN
            x_d(:,((Nlon_pseudo*nd)+1):(Nlon_pseudo*(1+nd))) = x(1+DelaySamplesMax-DelaySamples(nd):end-DelaySamples(nd),:);
            CoVariateFlat_d(:,((Nlon_pseudo*nd)+1):(Nlon_pseudo*(1+nd))) = x(1+DelaySamplesMax-DelaySamples(nd):end-DelaySamples(nd),:);
        end
        x = x_d; y = y_d; clear x_d; clear y_d;
        CoVariateFlat = CoVariateFlat_d; Psi = Psi_d; clear CoVariateFlat_d; clear Psi_d;
        [Nsamps,Nlon_pseudo] = size(x);
    else
        error('Not ready for DelaySamples<0')
    end
end


%% Separate time-samples to training and validation parts

TrainingSamples = 1:ceil(Nsamps*TrainValidTestFracs(1)); NsampsTrain = length(TrainingSamples);
if length(TrainValidTestFracs)==3
    ValidationSamples = TrainingSamples(end) + (1:ceil(Nsamps*TrainValidTestFracs(2))); %(144)
    TestSamples = (ValidationSamples(end)+1):Nsamps;
    ValidationAndTestSamples = [ValidationSamples,TestSamples];
elseif length(TrainValidTestFracs)==2 %i.e., no validation data in trainbr - make a "validation" set just for the uniformity of the script
    ValidationSamples = (TrainingSamples(end) + 1):Nsamps; 
    TestSamples = ValidationSamples; ValidationAndTestSamples = ValidationSamples;
else
    error('Incorrect TrainValidTestFracs length');
end
y1 = y(TrainingSamples); y2 = y(ValidationSamples); y3 = y(TestSamples); 
y1_raw = Psi(TrainingSamples); y2_raw = Psi(ValidationSamples); y3_raw = Psi(TestSamples);
x1 = x(TrainingSamples,:); x2 = x(ValidationSamples,:); x3 = x(TestSamples,:);
x1_raw = CoVariateFlat(TrainingSamples,:); 
x2_raw = CoVariateFlat(ValidationSamples,:); 
x3_raw = CoVariateFlat(TestSamples,:);
Years = (1:Nsamps)/12;

%% Linear Regression
if LinReg==1
    IncludeIntercept = 1;
    if IncludeIntercept==1
        x_r = [x,ones(Nsamps,1)]; x1_r = [x1,ones(length(TrainingSamples),1)]; x2_r = [x2,ones(length(ValidationSamples),1)]; 
    else
        x_r = x; x1_r = x1; x2_r = x2;
    end
    [b,bint,res,rint,stats] = regress(y1,x1_r);
    yp = sum(b.*x_r',1); y1p = sum(b.*x1_r',1); y2p = sum(b.*x2_r',1);

    cc = corrcoef([y2],yp(ValidationAndTestSamples')); cc = cc(1,2);
    
    fh = figure; plot(Years,y);hold on; plot(Years,yp); legend('Psi','Lin Regression');
    title(['Linear reg. Valid corr=',num2str(cc)]);
    figfn = [OutputFolder,'Regress_Moc',PsiStr,'And',CoVariateNamesStr,FnFiltTime,FnFiltLon,'_LinReg.png'];
    print(fh,figfn,'-dpng','-r0'); close(fh);
    
    MatFN = [OutputFolder,'Regress_Moc',PsiStr,'And',CoVariateNamesStr,FnFiltTime,FnFiltLon,'_LinReg.mat'];
    save(MatFN,'TrainingSamples','ValidationSamples','CoVariateNamesStr','cc',...
        'PsiMethod','latrange','lat0','FiltType','FiltWind','lat1','lat2',...
        'ZonalSmoothRadDeg','FiltType','LPF_Freq','LonsNotNan','IndsNotNanX','LonDec',...
        'y1','y2','y1p','y2p','y','yp','EccoYear0','Years');
end

%% Linear regression of gradient
if LinRegGrad==1
    ZonalBinning = 1;
    for CG=1%4:2:12%[1,2:2:12]
        if ZonalBinning==1
    %         CG = 4; 
            x1_b = mean(reshape(x1,[NsampsTrain,Nlon2/CG,CG]),3);
%             x1_b = mean(reshape(x1(:,1:Nlon2*NCovarsVec),[NsampsTrain,Nlon2*NCovarsVec/CG,CG]),3);
%             x1_b = [x1_b,x1(:,Nlon2*NCovarsVec+1:end)];

            x1_b2 = [x1_b(:,end),x1_b,x1_b(:,1)]; %To impose periodicity in gradient calc in next line
            x1_b = 0.5* ( x1_b2(:,3:end) - x1_b2(:,1:end-2) ); %Zonal difference operator
%             x1_b = x1; %!!!!!!!!1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            x1_b = [x1_b,ones([NsampsTrain,1])]; % Add a vector of ones for a constant intercept term in the regression
            [b,bint,res,rint,stats] = regress(y1,x1_b);
            lambda = [1,1e2,1e3,1e4,1e+05,1e6];
            [b_lasso,LassoFitInfo] = lasso(x1_b,y1,'Lambda',lambda); 
%             lassoPlot(b_lasso,LassoFitInfo,'PlotType','Lambda');

        %     b = b(1:end-1);b = movmean(repmat(b,[CG,1]),CG+1);
            x_b = mean(reshape(x,[Nsamps,Nlon2/CG,CG]),3); 
            x_b2 = [x_b(:,end),x_b,x_b(:,1)]; %To impose periodicity in gradient calc in next line
            x_b = 0.5* ( x_b2(:,3:end) - x_b2(:,1:end-2) ); %Zonal difference operator
            x_b = [x_b,ones([Nsamps,1])]; % Add a vector of ones for a constant intercept term in the regression
            Psi_fit = sum(b.*x_b',1);
            Psi_fit_lasso = x_b*b_lasso;
        else
            [b_lasso,LassoFitInfo] = lasso(x1,y1,'Lambda',0.1);
%             [b_lasso,LassoFitInfo] = lasso(x1,y1,'Lambda',0.1,'CV',10);
%             b_lasso = lasso(x1,y1,'Lambda',lambda); 
            Psi_fit = sum(b.*x_b',1);
            Psi_fit_lasso = x_b*b_lasso;
            Psi_fit_lasso = x*b_lasso;

            [b,bint,res,rint,stats] = regress(y1,x1);
            % Psi_fit = sum(b.*CoVariateFlat,1);
            Psi_fit = sum(b.*x',1);
        end
        cc = corrcoef(Psi(ValidationAndTestSamples),[y2;y3]); cc = cc(1,2);
        cc_lasso = corrcoef(Psi_fit_lasso(ValidationAndTestSamples),[y2;y3]); cc_lasso = cc_lasso(1,2);
        fh = figure; plot(Years,y);hold on;plot(Years,Psi_fit_lasso); 
           legend('Psi','Fit-Lasso'); 
        title(['LASSO Linear reg of zonal-grad. CG=',num2str(CG),'. lambda=',num2str(lambda(nlam)),...
        '. Lasso-Valid corr=',num2str(cc_lasso)]);   
        title(['Linear reg of zonal-grad. CG=',num2str(CG),'. Valid corr=',num2str(cc),...
        '. Lasso-Valid corr=',num2str(cc_lasso)]);
        figfn = [OutputFolder,'Regress_Moc',PsiStr,'And',CoVariateNamesStr,FnFiltTime,FnFiltLon,'_LASSO_lambda',num2str(lambda),'.png'];
        print(fh,figfn,'-dpng','-r0'); %close(fh);

        
    end
end

%% Robust linear regression

if LinRegRobust==1
    CG = 4; 
    x1_degraded = mean(reshape(x1,[length(y1),Nlon2/CG,CG]),3);
    [b,stats] = robustfit(x1_degraded,y1);
    b2 = movmean(repmat(b(2:end),[CG,1]),CG+1);

    Psi_fit = b(1)+sum(b2.*x',1);
    figure; plot(y);hold on; plot(Psi_fit); legend('Psi','Fit');
end

%% Perform stepwise regression
if 0
    mdl = stepwiselm(x1,y1);
end

%% create a feed-forward network with hidden layers specified by NneuronsList
R2_training = nan(length(NneuronsList),NNrepeats); mse_training = R2_training; corrcoef_training = R2_training;
R2_validation = nan(length(NneuronsList),NNrepeats); mse_validation = R2_validation; corrcoef_validation = R2_validation;
R2_testing = nan(length(NneuronsList),NNrepeats); mse_testing = R2_testing; corrcoef_testing = R2_testing;
mse_goal = nmse_goal*var(y(ValidationAndTestSamples));
for NneurN=1:length(NneuronsList)
    Nneurons = NneuronsList(NneurN);
    StrNneurons = num2str(Nneurons(1));
    StrNneurons = [num2str(NNNlayers),'x',num2str(Nneurons(1))]; 
    if length(Nneurons)>1
        for n=2:length(Nneurons)
            StrNneurons = [StrNneurons,'-',num2str(Nneurons(n))];
        end
    end
    if rng_seed>=0
        rng(rng_seed);
    end
    for nrep=1:NNrepeats
        % Nneurons = 6;
        % Nneurons = [10,8,5];
    %     rng(0);
        if LoadPrevNN==1
            load(FnPrevNN,'net');
        else
            clear net;
            for nnln = 1:NNNlayers; NNlayers(nnln) = Nneurons; end
            net = feedforwardnet(NNlayers);
            net.trainFcn = trainFcn; 
            net.performFcn = 'mse'; %That's the default, but let's be clear.
            if nmse_goal>0; net.trainParam.goal = mse_goal; end
            if Reg>0; net.performParam.regularization = Reg; end
            if or( strcmp(trainFcn,'trainlm'), strcmp(trainFcn,'trainbr') )
                if ~isempty(mu); net.trainParam.mu = mu; net.trainParam.mu_dec = mu_dec; net.trainParam.mu_inc = mu_inc; end
            end
            for nnln = 1:NNNlayers; net.layers{nnln}.transferFcn = ActivFunc; end %Set activation funciton
            if NetNorm=='Global' %Set input normzliation method
                net.inputs{1}.processFcns  = {}; net.outputs{2}.processFcns = {};
            elseif NetNorm=='mapstd'
                net.inputs{1}.processFcns={'mapstd'}; net.outputs{NNNlayers+1}.processFcns={'mapstd'};
            end
            net = configure(net,x',y');
            net.divideFcn = divideFcn; 
            if strcmp(divideFcn,'divideind')
                net.divideParam.trainInd = TrainingSamples;
                if length(TrainValidTestFracs)==3
                    net.divideParam.valInd   = ValidationSamples;
                end
                net.divideParam.testInd  = TestSamples;
            end
            net.trainParam.showWindow = TrainNN_Vis;% If ==0 -> disable NN training window popup
            net = train(net,x',y');
        end
        
        y1p = net(x1'); y2p_raw = net(x2_raw');
        y2p = net(x2'); y3p = net(x3'); yp = net(x');

%         corrcoef_validation_0 = corrcoef(y,yp)
        cc = corrcoef(y,yp);
        if DispFits; disp(['Correlation of complete y_filt vs NN(trained on x_filt) predicton from x_filt=',num2str(cc(1,2))]); end
        cc = corrcoef(y2_raw,y2p_raw);
        if DispFits; disp(['Correlation of raw y timeseries vs NN(trained on x_filt) predicton from raw x=',num2str(cc(1,2))]); end
        cc = corrcoef(y1,y1p); corrcoef_training(NneurN,nrep) = cc(1,2);
        SumSqRes = sum((y1-y1p').^2);
        R2_training(NneurN,nrep) = 1 - SumSqRes/sum((y1-mean(y1)).^2);
        mse_training(NneurN,nrep) = sqrt(SumSqRes/length(y1))/1e6;
        if DispFits; disp(['Training correlation of  y_filt vs NN(trained on x_filt) predicton from x_filt=',num2str(cc(1,2))]); end
        cc = corrcoef(y2,y2p); corrcoef_validation(NneurN,nrep) = cc(1,2);
        SumSqRes = sum((y2-y2p').^2);
        R2_validation(NneurN,nrep) = 1 - SumSqRes/sum((y2-mean(y2)).^2);
        mse_validation(NneurN,nrep) = sqrt(SumSqRes/length(y2))/1e6;
        if DispFits; disp(['Validation correlation of  y_filt vs NN(trained on x_filt) predicton from x_filt=',num2str(cc(1,2))]); end
        cc = corrcoef(y3,y3p); corrcoef_testing(NneurN,nrep) = cc(1,2);
        SumSqRes = sum((y3-y3p').^2);
        R2_testing(NneurN,nrep) = 1 - SumSqRes/sum((y3-mean(y3)).^2);
        mse_testing(NneurN,nrep) = sqrt(SumSqRes/length(y3))/1e6;
        if DispFits; disp(['Testing correlation of  y_filt vs NN(trained on x_filt) predicton from x_filt=',num2str(cc(1,2))]); end
        
        
        %         corrcoef_training(NneurN,nrep) = corrcoef_training_0(1,2);
        
%         corrcoef_training_0 = corrcoef(y1,y1t); 
%         corrcoef_training(NneurN,nrep) = corrcoef_training_0(1,2);
%         corrcoef_validation_0 = corrcoef(Psi(ValidationSamples),y2_raw);
%         corrcoef_validation_0 = corrcoef(Psi(ValidationSamples),y2(ValidationSamples));
%         

        %% Plot

        if NNrepeats>1
            nrepstr = ['_rep',num2str(nrep)];
        else
            nrepstr = '';
        end
%         fh = figure; plot(Years,Psi/10^6);hold on; plot(Years,y2/10^6); legend('Psi','Fit');
        fh = figure; plot(EccoYear0+Years,y/10^6); xlim([EccoYear0+Years(1),EccoYear0+Years(end)]);
        hold on; plot(EccoYear0+Years,yp/10^6); legend('Psi','Fit');
%         fh = figure; plot(EccoYear0+Years,y/10^6);hold on; plot(EccoYear0+Years,yp*std(y)/std(yp)/10^6); legend('Psi','Fit');
%         fh = figure; plot(EccoYear0+Years,y/10^6);hold on; plot(EccoYear0+Years,yp*9.8/10^6); legend('Psi','Fit');
        xlabel('Time [years]'); ylabel('Transport anomaly [Sv]'); 
        title({[BasinStrTitle,BotttomMOCstrTitle,PsiStr,' vs ',CoVariateNamesStr],[TitleFiltTime(3:end),TitleFiltLon],...
            ['Neural network fit. #neurons=',StrNneurons,'. Repeat#',num2str(nrep)],...
            ['Validation r^2=',num2str(corrcoef_validation(NneurN,nrep)^2),'. Testing r^2=',num2str(corrcoef_testing(NneurN,nrep)^2)],...
            ['Validation mse=',num2str(mse_validation(NneurN,nrep)),'. Testing mse=',num2str(mse_testing(NneurN,nrep)),' Sv'],...
            ['Validation R^2=',num2str(R2_validation(NneurN,nrep)),'. Testing R^2=',num2str(R2_testing(NneurN,nrep))]});
        figfn = [OutputFolder,'Regress_Moc',PsiStr,'And',CoVariateNamesStr,FnFiltTime,FnFiltLon,'_NN',StrNneurons,nrepstr,'.png'];
        print(fh,figfn,'-dpng','-r0'); close(fh);
        
        if and(strcmp(ActivFunc,'purelin'),length(CoVariateNames)==1)
            IW = net.IW{1};
            if length(IW)<length(lon)
                IW2 = nan*ones(size(lon));
                for n=1:Nlon2
                    IW2(IndsNotNanX(n)) = IW(n);
                end
                IW = IW2; clear IW2;
            end                
            fh = figure('Position',[50,50,1000,400]);
            yyaxis left; plot(lon,IW); ylabel('Weights');
            yyaxis right; plot(lon,-bath); ylabel('Bathymetry [m]');
            xlabel('Longitude'); grid;
            title({[BasinStrTitle,BotttomMOCstrTitle,PsiStr,' vs ',CoVariateNamesStr],[TitleFiltTime(3:end),TitleFiltLon],...
                ['Neural network fit. #neurons=',StrNneurons,'. Repeat#',num2str(nrep)],...
                ['Linear NN weights (left) and bathymetry (right)']});
            figfn = [OutputFolder,'Regress_Moc',PsiStr,'And',CoVariateNamesStr,FnFiltTime,FnFiltLon,'_NN',StrNneurons,nrepstr,'_WeightsVsBath.png'];
            print(fh,figfn,'-dpng','-r0'); close(fh);
        end


%         b1 = [0,0.4470,0.7410]; r1=[0.8500,0.3250,0.0980];
%         fh = figure; plot(EccoYear0+Years,y/10^6,'color',b1);%'b'); 
%         hold on; 
%         plot(EccoYear0+Years,Psi_fit_Lin/10^6,'--','color',[0,0,0]+0.51);
%         plot(EccoYear0+Years,yp/10^6,'color',r1);%'r'); 
%         xlim([EccoYear0+Years(1),EccoYear0+Years(end)]); ylim([-15,15]); grid;
%         ly = -15:0.01:15; 
%         lx = EccoYear0 + (Years(end)-Years(1))*TrainValidTestFracs(1); lx = lx*ones(size(ly)); plot(lx,ly,'k');
%         lx = EccoYear0 + (Years(end)-Years(1))*sum(TrainValidTestFracs(1:2)); lx = lx*ones(size(ly)); plot(lx,ly,'k');
%         xlabel('Time [years]'); ylabel('Transport [10^6 m^3/s]'); 
%         title({['Southern Ocean Deep Meridional Overturning Circulation']});
%         legend({'Raw data','Linear Fit','NN fit'},'Location','Southwest');
%         figfn = [OutputFolder,'Regress_Moc',PsiStr,'And',CoVariateNamesStr,FnFiltTime,FnFiltLon,'_NN',StrNneurons,nrepstr,'.png'];
%         print(fh,figfn,'-dpng','-r0'); close(fh);
% 
% 
            % cmap = cmocean('curl'); 
        % fh = figure; colorbar;
        % pcolor(Lon,Lat,b_reshaped); colormap(cmap);
        % rmax = max(abs(b_reshaped(:))); cc = linspace(-rmax,rmax,20);
        % shading flat; cb = colorbar; caxis([cc(1),cc(end)]);%cb=gcmfaces_cmap_cbar(cc);
        % xlabel('longitude'); ylabel('latitude');
        % % title([TitleCorrDesc,'Correlation between ',CoVariateNamesStr,' and PSI',num2str(PsiMethod),TitleFiltTime]);

        %% Save mat file
%         EccoYear0 = 1992; YearsECCO=EccoYear0+Years;
%         save('NN20_unfilt_smoothrad2.mat','net','YearsECCO','Psi');
        
        MatFN = [OutputFolder,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,...
            '_NN',StrNneurons,nrepstr,'_',trainFcn,'.mat'];
        save(MatFN,'net','Nneurons','TrainingSamples','ValidationSamples','CoVariateNamesStr','corrcoef_training',...
            'corrcoef_validation','corrcoef_testing','PsiMethod','latrange','lat0','FiltType','FiltWind','lat1','lat2',...
            'ZonalSmoothRadDeg','FiltType','LPF_Freq','LonsNotNan','IndsNotNanX','LonDec',...
            'y1','y2','y3','y1p','y2p','y3p','y','yp','EccoYear0','Years');
%         save(MatFN,'x','TrainingSamples', 'ValidationSamples', 'TestSamples','-append');
    end
    if NNrepeats>1
        MatFN = [OutputFolder,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,...
            '_NN',StrNneurons,'_','_',trainFcn,'_RepeatMatrix',num2str(NNrepeats),'.mat'];
        save(MatFN,'ECCOv','NNrepeats','NneuronsList','TrainingSamples','ValidationSamples','CoVariateNamesStr',...
            'corrcoef_training','corrcoef_validation','corrcoef_testing',...
            'PsiMethod','latrange','lat0','FiltType','FiltWind','lat1','lat2',...
            'ZonalSmoothRadDeg','FiltType','LPF_Freq','LonsNotNan','IndsNotNanX',...
            'R2_training','R2_validation','R2_testing',...
            'mse_training','mse_validation','mse_testing','LonDec');
    end
end

if length(NneuronsList)*length(NNrepeats)>1
    [NneuronsMat,RepsMat] = ndgrid(NneuronsList,1:NNrepeats);

    fh = figure; imagesc(corrcoef_validation); colorbar; 
    yticks([1:length(NneuronsList)]); xticks([1:NNrepeats]);  yticklabs = {}; 
    for n=1:length(NneuronsList); yticklabs{n} = num2str(NneuronsList(n)); end; yticklabels(yticklabs);
    title({[BasinStrTitle, BotttomMOCstrTitle,PsiStr,' vs ',CoVariateNamesStr],[TitleFiltTime(3:end),TitleFiltLon],...
        ['Neural network fits: validation set correlation']}); xlabel('repeat #'); ylabel('# Neurons');
    figfn = [OutputFolder,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,...
    'NN_RepeatMatrix',num2str(NNrepeats),'_MatValidation.png'];
    print(fh,figfn,'-dpng','-r0'); 

    fh = figure; imagesc(corrcoef_testing); colorbar; 
    yticks([1:length(NneuronsList)]); xticks([1:NNrepeats]);  yticklabs = {}; 
    for n=1:length(NneuronsList); yticklabs{n} = num2str(NneuronsList(n)); end; yticklabels(yticklabs);
    title({[BasinStrTitle, BotttomMOCstrTitle,PsiStr,' vs ',CoVariateNamesStr],[TitleFiltTime(3:end),TitleFiltLon],...
        ['Neural network fits: testing set correlation']}); xlabel('repeat #'); ylabel('# Neurons');
    figfn = [OutputFolder,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,...
    'NN_RepeatMatrix',num2str(NNrepeats),'_MatTesting.png'];
    print(fh,figfn,'-dpng','-r0'); 

    %Find a good minimal ylim value for both the validation and testing
    %plots, for easy comparison
    Rmin4plot = min(min(mean(corrcoef_validation,2)),min(mean(corrcoef_testing,2)));
    Rmin4plot = Round2Prec_V1(Rmin4plot,1,'floor');
       
    fh = figure; errorbar(NneuronsList,mean(corrcoef_validation,2),std(corrcoef_validation,[],2),'LineWidth',1); 
    hold on; errorbar(NneuronsList,mean(corrcoef_testing,2),std(corrcoef_testing,[],2),'LineWidth',1); 
    legend({'Validation','Testing'},'Location','Best');
    xlabel('# neurons'); ylabel('Correlation'); grid; ylim([Rmin4plot,1]);
    title({[BasinStrTitle, BotttomMOCstrTitle,PsiStr,' vs ',CoVariateNamesStr],[TitleFiltTime(3:end),TitleFiltLon],...
        ['Neural network fits #repeats=',num2str(NNrepeats),', correlation']});
    figfn = [OutputFolder,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,...
    'NN_RepeatMatrix',num2str(NNrepeats),'_TestingValidationCor.png'];
    print(fh,figfn,'-dpng','-r0'); 

    fh = figure; errorbar(NneuronsList,mean(mse_validation,2),std(mse_validation,[],2),'LineWidth',1); 
    hold on; errorbar(NneuronsList,mean(mse_testing,2),std(mse_testing,[],2),'LineWidth',1); 
    legend({'Validation','Testing'},'Location','Best');
    xlabel('# neurons'); ylabel('MSE [Sv]'); grid; 
    title({[BasinStrTitle, BotttomMOCstrTitle,PsiStr,' vs ',CoVariateNamesStr],[TitleFiltTime(3:end),TitleFiltLon],...
        ['Neural network fits #repeats=',num2str(NNrepeats),', MSE']});
    figfn = [OutputFolder,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,...
    'NN_RepeatMatrix',num2str(NNrepeats),'_TestingValidationMSE.png'];
    print(fh,figfn,'-dpng','-r0'); 
    
    fh = figure; errorbar(NneuronsList,mean(R2_validation,2),std(R2_validation,[],2),'LineWidth',1); 
    hold on; errorbar(NneuronsList,mean(R2_testing,2),std(R2_testing,[],2),'LineWidth',1); 
    legend({'Validation','Testing'},'Location','Best');
    xlabel('# neurons'); ylabel('R^2 [-]'); grid; 
    title({[BasinStrTitle, BotttomMOCstrTitle,PsiStr,' vs ',CoVariateNamesStr],[TitleFiltTime(3:end),TitleFiltLon],...
        ['Neural network fits #repeats=',num2str(NNrepeats),', coefficient of determination']});
    figfn = [OutputFolder,CoVariateNamesStr,'VsMoc',PsiStr,FnFiltTime,FnFiltLon,FnCorrDesc,...
    'NN_RepeatMatrix',num2str(NNrepeats),'_TestingValidationR2.png'];
    print(fh,figfn,'-dpng','-r0'); 
end



%%
% IW11=net.IW{1,1};
% nneur = 2; figure; yyaxis left; plot(lon,IW11(nneur,:)); yyaxis right; plot(lon,-bath); 
% % figure; plot(lon,gradient(movmean(IW11(1,:),5)));
% % figure; plot(lon,cumsum(IW11,2));%,:));
% 
% corrcoef(IW11(2,:),bath)

set(groot, 'DefaultFigureVisible', 'on');

end

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

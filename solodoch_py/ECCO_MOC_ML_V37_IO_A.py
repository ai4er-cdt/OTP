import os
import sys
import gsw
import math
import random
import matplotlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import statsmodels.api as sm

from datetime import datetime
from scipy.io import loadmat
from scipy.signal import gaussian, convolve, butter, lfilter
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression, LassoCV, HuberRegressor

from keras.models import Sequential
from keras.layers import Dense

import utils, LLC2ZonalStrip_V3, read_nctiles_V4r3r4

"""
-- Aviv 2021-02-15 -- 

 V4             add option to use ECCOV4r4 (instead of r3) daily fields (only monthly # exist in r3 case)
 V5 2021-11     add option to do either bottom or intermediate MOC cell
 V6 2021-11-29: a. Correct bugs in V5 regarding cells
                b. Add option to use multiple co-variates, e.g., SSH and BP
                c. Add rng seed setting for debugging
 V7 2021-12-01  add option for AMOC (rathern than all lons 0-360 deg)
 V8 2021-12-06  add ability to do multiple layer NNs
 V9 2021-12-13  make it operable on my Linux laptop
 V10 2021-12-14 add a trainFcn parameter
 V11 2021-12-15 a. Corrected bug in LLC2ZonalStrip_V1.m->V2
                b. Using basin masks (which I made in my MakeMasks_Va.m script today) to limit covariates to specific basins.
 V11 2021-12-19 add option to include zonal wind as a zonal integral only (omit zonal dependence), i.e., as integrated Sverdrap Transport
 
 -- 2022 --
 
 V12            a. Made a nonzero mse target
                b. Added a Coefficient of Determination metric in plots
                c. Added regularization option
 V13            add option for longitudinal decimation, to decrease NN size
 V14            a. add option to disable NN training window popup
                b. More flexible Decimation specification.
 V15            alow Indo-Pacific sector
 V16            allow delay-samples (poor-man's RNN).
 V16_IO 2022-03-12 allow arbitrary input variables for easy parameter sweeps with external-interface
 V17 2022-03-15 a. Add OutFoldParseByLat
                b. Save to mat file also the predictant (MOC, y) and NN predictions (output, yp) time series, 
                so you don't have to relaod and preprocesses input data anytime you want to analyze the output yp vs predictant y.
 V18 2022-03-22 a. Allow loading SST and SSS
                b. Allow OutFoldParseByDelay in addition to OutFoldParseByLat & OutFoldParseByFiltTime
 V19 2022-04-13 a. Add dlat to the output folder name
                b. Add option for "Extended" Atlantic Ocean mask (i.e., including Med Sea), since the AMOC time series was calculated on the extended basin.
 V20 2022-04-29 Change naming of basins. Basin==1 specifies MOC data and covariate masks corresponding to the Atlantic without Med.
 V21 2022-05-14 a. Name bug correctoin for trainbr cases with nmse_goal=0.
 V22 2022-05-23 a. Add Psi6 option: diagnose MOC by average value across a preseribed latitude range, on the density level giving max of the time-mean
                b. Simplify output folder naming conventions for Psi's on a latitude range
                c. correct bug in delayed samples option (ddid not work with multiple input vars).
 V23 2022-06-06 a. Corrected bug: forgot to detrend scalar (zonal-mean) covariates
                b. For 3 or less zonal points, detrend3 didn't work, so I loop and use detrend
                c. Corrected bug: covariates were not deseasoned (just detrended), while the MOC time series was. Adding switches to specify desired operations.
                d. Removed SAM lines from code.
 V33            a. Corrected errors with deseaoning and detrending of input X&Y
                b. Made output folder name dependent on choices for deseasnong and detrending.
 V34            output folder name dependent on training set size.
 V35            a. control Levenberg-Marquardt Algorithm convergence speed via mu parameters. Already did so in previous version, but not for trainbr (which uses trainlm). 
                b. Made mu params choices reflected in output filder name
 V36 2022-07-14 a. Corrected bug - zonal smoothing was done only after collapsing multiple covariates' zonal structures to one dimension
                b. Allowing non-consecutive time-delays
 V37 2022-08-08 Fix flags for chossing between NNs and linear regression
"""

HOMEDIR = "C:/Users/aline/OTP/ECCO4_Release3"

def fnos(fnin, os_name):
    if os_name == "Linux":
        if fnin.startswith("D:\\Aviv\\") or fnin.startswith("D:/Aviv/"):
            fnout = "/data/" + fnin[len("D:\\Aviv\\") :]
    else:
        fnout = fnin.replace("\\", "/")
    return fnout


# WHERE IS THE ACTUAL FUNCTION?
# def ECCO_AABW_Regress_ZonStrip_V37_IO_A(varargin):

OS = "Windows"
sys.path.append("/data/Research/MATLAB_Scripts/ClimateDataToolbox/")
sys.path.append("/data/Research/MATLAB_Scripts/Utilities/")
# Show values of MSE, correlation, etc
DispFits = 0

# Set the visibility of figures and training window popup (0 == disable any display)
FigVis = 0
if FigVis == 0:
    matplotlib.use("Agg")
TrainNN_Vis = 0  # If ==0 -> disable NN training window popup

# Define flags for output folder parsing
LoadPrev = 0
OutFoldParseByLat = 1
OutFoldParseByFiltTime = 0
OutFoldParseByDelay = 0

# Define flags for linear regression
LinReg = 0  # if LinReg==1, use linear regression, i.e., most of the following flags are not used
LinRegGrad = 0
LinRegRobust = 0

"""
Neural network training function and parameters
    * Network training options are -- 'trainbfg - traingdx - trainbr - trainlm - trainscg'
    * Mu controlls convergence rate (the higher, the slower)
    * Default mu values:
        initial mu = 0.005
        mu_dec(rease rate) = 0.1
        mu_inc(rease rate) = 10
        mu[] to use default values for all 3 params
    * Reg is a network cost function regularisation parameter
        reg = 0 corresponds to no regularization
    * nmse is the normalised MSE performance goal (a finite value can help the training process converge)
"""

trainFcn = "trainbr"
if trainFcn in ["trainlm", "trainbr"]:
    mu = 0.005
    mu_dec = 0.1
    mu_inc = 10
if trainFcn == "trainbr":
    Reg = 0
else:
    Reg = 0.75  # Regularization parameter

# Neural network performance goal (normalised performance goal) and activation function
nmse_goal = 0
ActivFunc = "relu"  # 'linear - tanh'

# Neural network normalization and data division function
NetNorm = "mapstd"  # 'minmax', 'Global'
divideFcn = "divideind"  # 'dividerand', 'divideint'

# Training, validation, and test set fractions
TrainValidTestFracs = [0.7, 0.3]  # [0.4, 0.3, 0.3]

# Random number generator seed
rng_seed = 1  # Set to -1 for no seed setting

# Neural network architecture
NNNlayers = 1  # Number of NN hidden layers
NneuronsList = [
    1,
    3,
    5,
]  # [1,5,10,15,20] - [1,3,5] - [1,5,10,15,20] - [20] - [5,10,15,20,25,30] - [3,5,8,10,15,20]
# [5,10,15,20,30] - [40,50,60] - [50,100] - [4,6,8,10,15,20]
NNrepeats = 5  # Number of repetitions for training
if LinReg == 1:
    NneuronsList = []

# ECCO version and year
ECCOv = "V4r3"  # 'V4r4';
EccoYear0 = 1992

"""
MOC cell type and basin
    * Basin = 1 for AMOC/Atl-basin
    * Basin = 1.5 for Atl covariates and Atl-Med AMOC
    * Basin = 1.75 for Atl+Med AMOC + covariates
    * Basin = 2 for PMOC / Indo-Pacific
    * Basin = 0 for all lons (0-360 degrees)
"""
BottomMOC = 1  # ==1 (0) for bottom (top) MOC cell
Basin = 2

"""
MOC time series definition method
    -- How to get a single MOC strenght per time sample from the MOC (lat, density) distribution
Covariate names and preprocessing flags
    -  {'ETAN','oceTAUX_int'}
    -  {'SIheff'}
    -  {'THETA_S','SALT_S'}
    -  {'OBP','ETAN','oceTAUX_int','THETA_S','SALT_S'}
    -  {'OBP','oceTAUX_int'}
    -  {'ETAN','oceTAUX_int'}
    -  {'oceTAUX_int'}  "_int" means zonal integral
    -  {{{'OBP','ETAN','oceTAUX'};# 'OBP'}'ETAN'}
    -  'oceTAUX' - 'DensFlux' - 'oceTAUY' - 'PHIBOT' - 'OBPNOPAB' - 'ETAN' - 'PHIBOT'
"""
PsiMethod = 5  # 4
CoVariateNames = ["OBP"]
DetrendCovar = 1
DeseasonCovar = 1
DetrendMOC = 2
DeseasonMOC = 2

"""
# Delay samples for covariates
    [3,6,9,12]; 
    DelaySamples = 0: only instantenous samples
    DelaySamples >=1: number of additional delays
    DelaySamples <=-1: (-#) of single-sample delay
"""
DelaySamples = 0

"""
Filtering type and parameters (Low-pass filter frequency)
    - 'LPF'
    - Frequency: 1/6 - 1/(2*12) - [1/month]
Zonal smoothing radius in degrees
    - 0 or 2
Longitudinal decimation factor -- decimate (subsample) zonal points of each covariate by this factor 
    - 1: no decimation is done
    - 10 or 20
Bathymetry threshold -- depth threshold for grid cell usability (m)
    - 0 or 150
"""
FiltType = ""
LPF_Freq = 1 / 18
ZonalSmoothRadDeg = 0
LonDec = 1
BathThresh = 0

"""
## DENSITY ##
Density separation value = [kg/m^3] isopycnal separating upper and lower MOC branches 
MOC cell latitude range and density value
    - BottomMOC == 1 is a Deep MOC cell 
        with options for Covariate:
        lat1 = -62 and lat2 = -58
        lat1 = -22 and lat2 = -18
    - Intermediate MOC cell 
        lat1 = lat0-dlat
        lat2 = lat0+dlat    
"""
dens_sep = 1036.99
if BottomMOC == 1:
    latrange = [-60, -40]
    lat0 = -60  # -50 for MOC time series
    if Basin == 0:
        DenseVal = 1037.1
    elif Basin == 2:
        DenseVal = 1037.05
    else:
        raise ValueError("Unspecified density for this basin")
    dlat = 2
    lat1 = latrange[0]
    lat2 = latrange[1]
else:
    if Basin == 1:
        DenseVal = 1036.8
    else:
        raise ValueError("Unspecified density for this basin")
    lat0 = 26  # latrange = [21,31] for MOC time series
    dlat = 1
    latrange = [lat0 - dlat, lat0 + dlat]
    lat1 = lat0 - dlat
    lat2 = lat0 + dlat

""" Use variable-length input variable list (in dictionary-like format):
        ECCO_AABW_Regress_ZonStrip_V16_IO({'PsiMethod',4},{'LonDec',2},{'FiltType','LPF'})
"""


def dynamic_assign(*args):
    for var_name, value in args:
        # Directly update the local scope of the caller with the new variable and its value.
        locals()[var_name] = value
    # e.g.  dynamic_assign(('var1', 10), ('var2', 'hello'))


"""
# ALTERNATIVE for dynamic assign:
for nv in range(len(varargin)):
    locals()[varargin[nv][0]] = varargin[nv][1]
    """

# A hack to make external parameter sweeps on LPF freq easier to handle
if LPF_Freq == 1:
    FiltType = ""

# If (==1) and applying decimation, treat first and last zonal points as ~identical.
if Basin > 0:
    LonDecPeriodic = 0
else:
    LonDecPeriodic = 1

CorrType = "Corr"  # 'PartialCorr' - 'LagCorr'
LoadPrevNN = 0
FnPrevNN = fnos(
    "D:/Aviv/Research/AABWvsACC/ECCO/MocRegression/ECCOV4r3/PHIBOTVsMocPsi4_LPF6_ZonSmoothSig2_NN10_rep1.mat",
    OS,
)
# FnPrevNN = 'D:/Aviv/Research/AABWvsACC/ECCO/MocCorrelates/PHIBOTVsMocPsi4_LPF6__NN10_rep1.mat';

if ECCOv == "V4r4":
    Year1 = 1992
    Year2 = 2016
    Month1 = 1
    Month2 = 12

    Years = list(range(Year1, Year2 + 1))
    Months = list(range(Month1, Month2 + 1))

    YearsVec = np.tile(Years, len(Months)).flatten()
    MonthsVec = np.repeat(Months, len(Years))

    Nsamps = (Year2 - Year1 + 1) * (Month2 - Month1 + 1)

    startdate = datetime.strptime("1992-01-01", "# Y-# m-# d").toordinal()
    enddate = datetime.strptime("2016-12-31", "# Y-# m-# d").toordinal()

    sys.path.append("/ECCOv4r4Andrew")  # Adjust the path as necessary
    dirv4r4 = fnos("D:/Research/NumericalModels/ECCO/Version4/Release4/", OS)

    if Basin == 0:
        if DetrendMOC * DeseasonMOC >= 1:
            FN_MOC_Inds = fnos(
                "D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/DeepSoMOCtot_deseas_detrend_Inds2d_ECCOV4r4.mat",
                OS,
            )
        else:
            raise ValueError("Where is Var4r4 raw MOC?")
    else:
        raise ValueError("V4r4 AMOC/PMOC Files do not exist yet")

else:
    Nsamps = 288

    dirv4r3 = fnos("D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/", OS)
    deseadetrens_str = ""

    # Constructing the string based on DeseasonMOC and DetrendMOC flags
    if DeseasonMOC >= 1:
        deseadetrens_str += "_deseas"
    if DetrendMOC >= 1:
        deseadetrens_str += "_detrend"

    if Basin == 0:
        FN_MOC_Inds = fnos(
            f"D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/DeepSoMOC{deseadetrens_str}_Inds.mat",
            OS,
        )
    else:
        if math.floor(Basin) == 1.5:
            FN_MOC_Inds = fnos(
                f"D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/AtlAndMed/AMOCMed{deseadetrens_str}_Inds.mat",
                OS,
            )
        elif math.floor(Basin) == 1:
            FN_MOC_Inds = fnos(
                f"D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/Atl/AMOC{deseadetrens_str}_Inds.mat",
                OS,
            )
        elif Basin == 2:
            FN_MOC_Inds = fnos(
                f"D:/Aviv/Research/AABWvsACC/ECCO/MocDiagnostics/Pac/IndoPac{deseadetrens_str}_Inds.mat",
                OS,
            )
        else:
            raise ValueError("No such basin option")

p = fnos("D:/Aviv/Research/MATLAB_Scripts/Ocean/gcmfaces/", OS)
dirGrid = fnos(os.path.join(HOMEDIR, 'nctiles_grid'), OS)

if Basin == 1:
    BasinMasksFN = fnos(
        "D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/BasinMasks.mat", OS
    )
else:
    BasinMasksFN = fnos(
        "D:/Aviv/Research/NumericalModels/ECCO/Version4/Release3/BasinMasksWithMed.mat",
        OS,
    )
FN_DensFlux = fnos(
    "D:/Aviv/Research/AABWvsACC/ECCO/MocCorrelates/ECCO4r3_SurfDensFlux.mat", OS
)

deseadetrens_str = ""

# Handling DetrendCovar and DeseasonCovar
if DetrendCovar or DeseasonCovar:
    deseadetrens_str = "_Xd"
    if DetrendCovar == 1:
        deseadetrens_str += "T"
    if DeseasonCovar == 1:
        deseadetrens_str += "S"

# Handling DetrendMOC and DeseasonMOC
if DetrendMOC or DeseasonMOC:
    deseadetrens_str += "_Yd"
    if DetrendMOC >= 1:
        deseadetrens_str += "T"
    if DetrendMOC > 1:
        deseadetrens_str += "2"
    if DeseasonMOC >= 1:
        deseadetrens_str += "S"
    if DeseasonMOC > 1:
        deseadetrens_str += "2"

# Constructing the OutputFolder path
OutputFolder = fnos(os.path.join(HOMEDIR, 'MocRegression', f"ECCO{ECCOv}/LatSpecific{deseadetrens_str}/Reps{NNrepeats}"),
    OS,
)
print(OutputFolder)

# Handling BottomMOC condition
if BottomMOC == 1:
    BottomMOCstr = "Deepcell_"
    BottomMOCstrTitle = "Deep cell "
else:
    BottomMOCstr = "Topcell_"
    BottomMOCstrTitle = "Top cell "

# Handling Basin condition
if Basin == 1:
    BasinStr = "Atl_"
    BasinStrTitle = "Atl "
elif Basin == 1.5:
    BasinStr = "AtlMed_"
    BasinStrTitle = "Atl and Med "
elif Basin == 1.75:
    BasinStr = "AtlMed2_"
    BasinStrTitle = "Atl and Med x2 "
elif Basin == 2:
    BasinStr = "IndoPac_"
    BasinStrTitle = "IndoPac "
else:
    BasinStr = ""
    BasinStrTitle = ""

nFaces = 5  # Number of faces in this GCM setup
omitNativeGrid = not any(
    fname.startswith("tile001.mitgrid") for fname in os.listdir(dirGrid)
)

mygrid = utils.grid_load(dirGrid)

# Handling Zonal Smooth Radius Degree
if ZonalSmoothRadDeg > 0:
    TitleFiltLon = f", {ZonalSmoothRadDeg}deg zonal smooth rad"
    FnFiltLon = f"_ZonSmoothSig{ZonalSmoothRadDeg}"
else:
    TitleFiltLon = ""
    FnFiltLon = ""

# Handling Correlation Type
if CorrType == "Corr":
    TitleCorrDesc = ""
    FnCorrDesc = "_"
elif CorrType == "LagCorr":
    TitleCorrDesc = "Lagged-"
    FnCorrDesc = "_Lagged"
elif CorrType == "PartialCorr":
    TitleCorrDesc = "Partial-"
    FnCorrDesc = "_Part"
else:
    raise ValueError(f"Unrecognized CorrType: {CorrType}")

# Handling Filter Type
if FiltType == "":
    TitleFiltTime = ""
    FnFiltTime = ""
    FiltWind = 0
elif FiltType == "LPF":
    # Assuming LPF_Freq is non-zero to avoid division by zero error
    if LPF_Freq == 0:
        raise ValueError("LPF_Freq cannot be zero.")
    TitleFiltTime = f", {1 / LPF_Freq}months LPF"
    FnFiltTime = f"_LPF{1 / LPF_Freq}"
    FiltWind = LPF_Freq
else:
    raise ValueError(f"Unrecognized FiltType: {FiltType}")

PsiStr = f"Psi{PsiMethod}"
if PsiMethod == 5:
    PsiStr += f"Dens{DenseVal}"

# Assuming lat0 is predefined
LatStr = f"lat{lat0}"

# Assuming CoVariateNames is a list of strings
NCovars = len(CoVariateNames)
CoVariateNamesStr = "+".join(CoVariateNames)

# Assuming rng_seed is predefined
if rng_seed >= 0:
    seedn = rng_seed
else:
    seedn = random.randint(0, 2**32 - 1)  # Simulate MATLAB's rng for a new seed
random.seed(seedn)  # Set the seed
seedn = str(seedn)

# Assuming Reg and nmse_goal are predefined
RegStr = f"_Reg{Reg}" if Reg > 0 else ""
NmseStr = f"_NMSE{nmse_goal}" if nmse_goal > 0 else ""

LonDecStr = ""
if LonDec != 1:
    if LonDec == 0:
        raise ValueError("LonDec=0")
    LonDecStr = f"_LonDec{LonDec}"
    try:
        LonDecStr += f"Start{DecimStartInd}"
    except NameError:
        pass  # Do nothing, or handle appropriately

# Assuming DelaySamples is a predefined list or int
if isinstance(DelaySamples, int):
    DelaySamples = [DelaySamples]

if any(sample != 0 for sample in DelaySamples):
    DelayStr = f"_Delay{DelaySamples[0]}"
    for nd in range(1, len(DelaySamples)):
        DelayStr += f"-{DelaySamples[nd]}"
else:
    DelayStr = ""

# Assuming dlat is predefined
dlatstr = f"_dlat{dlat}"

# Assuming TrainValidTestFracs is a list or array of fractions
IndsStr = "".join(str(frac) for frac in TrainValidTestFracs)
IndsStr = IndsStr.replace(".", "")

# Assuming OutFoldParseByLat, PsiStr, and LatStr are predefined
OutFoldSec = ""
if OutFoldParseByLat == 0:
    PsiStr += LatStr
else:
    OutFoldSec = LatStr

trainFcnStr = trainFcn
if trainFcn in ["trainlm", "trainbr"]:
    if mu:  # Checking if mu is not None and not empty
        trainFcnStr = f"{trainFcnStr}-mu{mu}-{mu_dec}-{mu_inc}"

if LinReg == 1:
    RegType = f"LinReg_{divideFcn}{IndsStr}{LonDecStr}{dlatstr}"
else:
    RegType = f"NN_{NNNlayers}layers_{ActivFunc}_{trainFcnStr}{RegStr}{NmseStr}_{NetNorm}_{divideFcn}{IndsStr}{LonDecStr}{dlatstr}"

if OutFoldParseByFiltTime == 1:
    OutFoldSec += FnFiltTime
    OutputFolder += f"{BasinStr}{BottomMOCstr}{CoVariateNamesStr}VsMoc{PsiStr}{FnFiltLon}{FnCorrDesc}{RegType}"
else:
    OutputFolder += f"{BasinStr}{BottomMOCstr}{CoVariateNamesStr}VsMoc{PsiStr}{FnFiltTime}{FnFiltLon}{FnCorrDesc}{RegType}"

if BathThresh > 0:
    OutputFolder += f"_Bath{BathThresh}"

if OutFoldParseByDelay == 0:
    OutputFolder += f"{DelayStr}_seed{seedn}/"  # Using forward slash directly for path, Python handles this well across platforms
else:
    OutputFolder += f"_seed{seedn}/"
    OutFoldSec += DelayStr

# Update OutputFolder with OutFoldSec if it's not empty
if OutFoldSec:
    OutputFolder = os.path.join(OutputFolder, OutFoldSec)

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(OutputFolder):
    os.makedirs(OutputFolder)

# Display the OutputFolder path
print("OutputFolder=", OutputFolder)

"""
Generate Lon-Lat grid
    - fileName= [dirv4r3 'nctiles_monthly/PHIBOT/' 'PHIBOT']
    - fldName='PHIBOT'
    - MONTH = 1
    - PHIBOT=read_nctiles(fileName,fldName,MONTH,1), need to plug in 1 for surface depth if its a 4d field
    - [Lon,Lat,~] = LLC2ZonalStrip_V3(mygrid,PHIBOT,lat1,lat2)
    
Basin
    - 1 for AMOC/Atl-basin
    - 1.5 for Atl+Med Sea
    - 2 for PMOC/Indo-Pacific
    - 0 for all lons (0-360 degrees)
"""
# Assuming mygrid is defined and lat1, lat2 are the latitude bounds
Lon, Lat, Bath = LLC2ZonalStrip_V3.LLC2ZonalStrip_V3(mygrid, mygrid.Depth, lat1, lat2)

Nlat, Nlon = Lon.shape
lon = Lon[0, :]
bath = Bath.mean(axis=0)

if BathThresh > 0 or Basin > 0:
    CoVariateMask = np.ones((Nlat, Nlon))

    # Load basin mask
    if Basin in [
        1,
        1.5,
        2,
    ]:  # Assuming BasinMasksFN is defined and points to the correct file
        basin_data = loadmat(BasinMasksFN)
        X, Y = basin_data["X"], basin_data["Y"]
        if Basin == 1:
            MaskBasin = basin_data["MaskAtl"]
        elif Basin == 2:
            MaskBasin = basin_data["MaskPac"]

    if Basin > 0:
        for nlat in range(Nlat):
            lat = np.unique(Lat[nlat, :])
            ny, nx = np.where(Y == lat)
            if len(np.unique(ny)) != 1:
                raise ValueError("You are in a non-cartesian part of grid")
            else:
                ny = np.unique(ny)
                nx1, nx2 = np.searchsorted(X[:, ny], [lon.min(), lon.max()])
                CoVariateMask[nlat, :] = MaskBasin[nx1:nx2, ny].squeeze()

    if BathThresh > 0:
        CoVariateMask[Bath < BathThresh] = 0

    CoVariateMask[CoVariateMask != 1] = 0
    nx = np.where(CoVariateMask == 1)[1]
    mask_nx = np.unique(nx)
    CoVariateMask = CoVariateMask[:, mask_nx]
    CoVariateMask[CoVariateMask == 0] = np.nan
    Lon, Lat = Lon[:, mask_nx], Lat[:, mask_nx]
    lon, bath = lon[mask_nx], bath[mask_nx]
    Nlat, Nlon = Lon.shape

# Identify and adjust CoVariateNames for zonal integrals
DoZonalIntegral = np.zeros(NCovars)
for nc, covname in enumerate(CoVariateNames):
    if len(covname) > 4 and covname.endswith("_int"):
        DoZonalIntegral[nc] = 1
        CoVariateNames[nc] = covname[:-4]

NumInts = int(np.sum(DoZonalIntegral))
NCovarsVec = NCovars - NumInts

# Initialize CoVariate and CoVariateScalars arrays
CoVariate = np.nan * np.zeros((Nlat, Nlon, NCovarsVec, Nsamps))
CoVariateScalars = np.nan * np.zeros((NumInts, Nsamps))

# Load and assign data if 'DensFlux' is in CoVariateNames
if "DensFlux" in CoVariateNames:
    nc = CoVariateNames.index("DensFlux")
    if (
        LoadPrev == 1
    ):  # Assuming LoadPrev is defined; replace isfile check with os.path.isfile in Python
        # Assuming FN_DensFlux is defined and points to the file
        # Replace the following MATLAB load with Python equivalent, e.g., using np.load, pd.read_csv, or xr.open_dataset based on file format
        alldensflux = xr.open_dataset(FN_DensFlux)  # Load your data here
        CoVariate[:, :, nc, :] = alldensflux
        del alldensflux
        # Need to plug in 1 for surface depth if its a 4d field
        A = read_nctiles_V4r3r4.read_nctiles(
            [dirv4r3, "nctiles_monthly/PHIBOT/PHIBOT"], "PHIBOT", 1, 1
        )
        [X, Y] = utils.convert2pcol(mygrid.XC, mygrid.YC, A)  # [X, Y]
        del A


if LoadPrev == 0:
    for nt in range(Nsamps):
        nc_vecs = 0
        nc_scals = 0
        for nc in range(NCovars):
            CoVariateName = CoVariateNames[nc]
            CoVariate_t = None

            if CoVariateName == "oceTAUX":
                fldx = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/oceTAUX/oceTAUX.{nt:04d}.nc"
                )["oceTAUX"]
                fldy = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/oceTAUY/oceTAUY.{nt:04d}.nc"
                )["oceTAUY"]
                fldUe, fldVn = utils.calc_UEVNfromUXVY(mygrid, fldx.values, fldy.values)
                CoVariate_t = LLC2ZonalStrip_V3(mygrid, fldUe, lat1, lat2)
            elif CoVariateName == "oceTAUY":
                fldx = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/oceTAUX/oceTAUX.{nt:04d}.nc"
                )["oceTAUX"]
                fldy = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/oceTAUY/oceTAUY.{nt:04d}.nc"
                )["oceTAUY"]
                fldUe, fldVn = calc_UEVNfromUXVY(fldx.values, fldy.values)
                CoVariate_t = LLC2ZonalStrip_V3(mygrid, fldVn, lat1, lat2)
            elif CoVariateName == "DensFlux":
                SALT = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/SALT_S/SALT_S.{nt:04d}.nc"
                )["SALT"]
                THETA = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/THETA_S/THETA_S.{nt:04d}.nc"
                )["THETA"]
                SFLUX = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/SFLUX/SFLUX.{nt:04d}.nc"
                )["SFLUX"]
                TFLUX = xr.open_dataset(
                    f"dirv4r3/nctiles_monthly/TFLUX/TFLUX.{nt:04d}.nc"
                )["TFLUX"]
                Press_db = 10 * np.ones(SALT.shape)  # [db]
                dens = gsw.rho(SALT.values, THETA.values, Press_db)
                Cp = gsw.enthalpy(SALT.values, THETA.values, Press_db)
                alpha = gsw.alpha(
                    SALT.values, THETA.values, Press_db
                )  # keyword = 'ptmp'
                beta = gsw.beta(SALT.values, THETA.values, Press_db)  # keyword= 'ptmp'
                CoVariate_t = beta * SFLUX.values - alpha * TFLUX.values / Cp
            else:
                if ECCOv == "V4r3":
                    file_name = f"dirv4r3/nctiles_monthly/{CoVariateName}/{CoVariateName}.{nt:04d}.nc"
                else:
                    year = str(YearsVec[nt])
                    month = str(MonthsVec[nt])
                    file_name = f"dirv4r4/nctiles_monthly/{CoVariateName}/{CoVariateName}_{year}_{month}.nc"
                ds = xr.open_dataset(file_name)
                CoVariate_t = read_nctiles_V4r3r4(
                    file_name, CoVariateName, nt, 1, ECCOv
                )
                CoVariate_t = LLC2ZonalStrip_V3(mygrid, CoVariate_t, lat1, lat2)

            if CoVariateMask is not None:
                CoVariate_t = CoVariate_t[:, mask_nx] * CoVariateMask

            # This would include the full zonal dependence
            if DoZonalIntegral[nc] == 0:
                nc_vecs += 1
                CoVariate[:, :, nc_vecs, nt] = CoVariate_t.values
            # This variable is to become a zonal-mean sum
            else:
                nc_scals += 1
                if CoVariateName == "oceTAUX":
                    f = 2 * np.pi * 2 / 24 / 3600 * np.sin(np.deg2rad(lat0))  # [rad/s]
                    rho0 = 1025  # [kg/m^3]
                    Re = 6.378e6  # [m] Earth radius
                    CoVariateScalars[nc_scals, nt] = (
                        -np.nansum(CoVariate_t)
                        * (np.pi / 180)
                        * Re
                        * np.cos(np.deg2rad(lat0))
                        / (f * rho0)
                    )
                else:
                    CoVariateScalars[nc, nt] = np.nansum(CoVariate_t)
    if "DensFlux" in CoVariateNames:
        nc = CoVariateNames.index("DensFlux")
        alldensflux = CoVariate[:, :, nc, :]
        np.save(FN_DensFlux, alldensflux)
        del alldensflux

## MOC CALCULATION ##

"""
# TO DO - check format
data = np.load('FN_MOC_Inds.npy')
Psi1 = data['Psi1']
"""


if ECCOv == "V4r4":
    print(
        "(Aviv) Warning, in V5 I have not taken care of upper cell/lower cell option in EccoV4r4"
    )

    if PsiMethod == 1:
        Psi_d = FN_MOC_Inds.index("Psi1")
    elif PsiMethod == 2:
        Psi_d = FN_MOC_Inds.index("Psi2")
    elif PsiMethod == 3:
        Psi_d = FN_MOC_Inds.index("Psi3")
    elif PsiMethod == 4:
        Psi_d = FN_MOC_Inds.index("Psi4")

    Ndays = len(Psi_d)
    caldays = []
    for i in range(Ndays):
        caldays.append(startdate)
        startdate = startdate.replace(day=startdate.day + 1)
    y1 = int(str(caldays[0])[-4:])
    y2 = int(str(caldays[-1])[-4:])
    Nyears = y2 - y1 + 1

    if 0:
        caldays2 = [datetime.strptime(str(d), "%Y-%m-%d") for d in caldays]
        Psi_m, yr = reshapetimeseries(caldays2, Psi_d, "bin", "month")
        Psi = np.reshape(Psi_m, [1, 12 * Nyears])
    else:
        Psi = np.zeros([1, 12 * Nyears])
        PsiCum = Psi_d[0]
        mon0 = str(caldays[0])[5:7]
        nmon = 1
        dayspermonth = 1

        for nd in range(1, Ndays):
            mon1 = str(caldays[nd])[5:7]

            if nd == Ndays - 1:
                dayspermonth += 1
                Psi[0, nmon - 1] = PsiCum / dayspermonth
            elif mon0 == mon1:
                dayspermonth += 1
                PsiCum += Psi_d[nd]
            else:
                Psi[0, nmon - 1] = PsiCum / dayspermonth
                nmon += 1
                dayspermonth = 1
                PsiCum = Psi_d[nd]

            mon0 = mon1

    Nsamps = len(Psi)
else:
    lat = FN_MOC_Inds.index("lat")
    PSI_notrend = FN_MOC_Inds.index("PSI_notrend")
    dens_bnds = FN_MOC_Inds.index("dens_bnds")

    latinds = np.where((lat >= latrange[0]) & (lat <= latrange[-1]))[0]
    nlat0 = np.where(lat == lat0)[0][0]

    if BottomMOC == 1:
        CellDensInds = np.where(dens_bnds > dens_sep)[0]
        PSI_notrend = -PSI_notrend  # Make transport values positive
    else:
        CellDensInds = np.where(dens_bnds < dens_sep)[0]

    _, _, Nsamps = PSI_notrend.shape

    if PsiMethod == 4:
        PSI_Lat0_Mean = np.mean(PSI_notrend[nlat0, CellDensInds, :], axis=2)
        DensIndMax = np.argmax(PSI_Lat0_Mean)  # Pick density where time-mean is minimal
    elif PsiMethod == 5:
        DensIndChoice = np.argmin(
            np.abs(dens_bnds - DenseVal)
        )  # Mean over lat range at a particular density value choice
    elif PsiMethod == 6:
        P = np.mean(
            PSI_notrend[latinds, CellDensInds, :], axis=(0, 2)
        )  # Mean over time & lat range
        DensIndMaxLatRange = np.argmax(P)
        Psi = np.zeros(Nsamps)

    for nt in range(Nsamps):
        if PsiMethod == 1:
            Psi[nt] = np.max(
                PSI_notrend[nlat0, CellDensInds, nt]
            )  # Max over density at a given lat
        elif PsiMethod == 2:
            Psi[nt] = np.max(
                np.max(PSI_notrend[latinds, CellDensInds, nt], axis=1)
            )  # Max over density and lat at a given lat range
        elif PsiMethod == 3:
            Psi[nt] = np.min(
                np.max(PSI_notrend[latinds, CellDensInds, nt], axis=1)
            )  # Max over density, min over lat, at a given lat
        elif PsiMethod == 4:
            Psi[nt] = PSI_notrend[
                nlat0, CellDensInds[DensIndMax], nt
            ]  # Pick lat, then pick density where time-mean is minimal
        elif PsiMethod == 5:
            Psi[nt] = np.mean(
                PSI_notrend[latinds, DensIndChoice, nt]
            )  # Mean over lat range at a particular density value choice
        elif PsiMethod == 6:
            Psi[nt] = np.mean(
                PSI_notrend[latinds, CellDensInds[DensIndMaxLatRange], nt]
            )  # Mean over lat range at density where lat & time-mean is minimal

# Average in latitude, remove nans -- original definition: CoVariate ~ (Nlat,Nlon,NCovars,Nsamps)
CoVariateFlat = np.nanmean(CoVariate, axis=0)
x0 = np.sum(np.sum(CoVariateFlat, axis=2), axis=1)
IndsNotNanX = np.where(~np.isnan(x0))[0]
CoVariateFlat = CoVariateFlat[IndsNotNanX, :, :]
LonsNotNan = lon[IndsNotNanX]
Nlon2 = len(LonsNotNan)


# Zonal smoothing, if there are covariates which were not zonally averaged
if ZonalSmoothRadDeg > 0 and CoVariateFlat.size > 0:
    WinL = 11
    alpha = ((WinL - 1) / 2) / ZonalSmoothRadDeg
    GaussFilter = gaussian(WinL, alpha)
    GaussFilter /= GaussFilter.sum()

    for nt in range(Nsamps):
        for nv in range(NCovarsVec):
            CoVariateFlat[:, nv, nt] = convolve(
                CoVariateFlat[:, nv, nt], GaussFilter, mode="same"
            )


# Decimate longitude
if LonDec > 1:
    if LonDecPeriodic == 0:
        nn = np.arange(0, Nlon2, LonDec, dtype=int)
    else:
        nn = np.arange(
            0, Nlon2 - LonDec, LonDec, dtype=int
        )  # Periodic strip -> skip the last point

    LonsNotNan = LonsNotNan[nn]
    Nlon2 = len(LonsNotNan)
    CoVariateFlat = CoVariateFlat[nn, :, :]
elif LonDec < 0:  # In this case, (-)LonDec is the number of points to keep
    if LonDecPeriodic == 0:
        nn = np.linspace(0, Nlon2 - 1, -LonDec, dtype=int).round()
    else:
        nn = np.linspace(0, Nlon2 - 1, -LonDec + 1, dtype=int).round()
        if "DecimStartInd" in locals():
            nn = np.mod(nn + DecimStartInd, Nlon2)
            nn[nn == 0] = Nlon2
        nn = nn[:-1]  # Periodic strip -> throw away the last point

    LonsNotNan = LonsNotNan[nn]
    Nlon2 = len(LonsNotNan)
    CoVariateFlat = CoVariateFlat[nn, :, :]


# Create an array of datetime objects
dates = [datetime.datetime(Y[i], M[i], round(D[i])) for i in range(Nsamps)]
Psi = np.array(Psi)

# Detrend the data
if DetrendMOC == 2:
    Psi = detrend(Psi)

# Deseason the data
if DeseasonMOC == 2:
    # Assuming Psi is a univariate time series
    decomposition = sm.tsa.seasonal_decompose(
        Psi, model="additive", freq=12
    )  # Adjust freq as needed
    seasonal_component = decomposition.seasonal
    Psi = Psi - seasonal_component


# Detrending and deseasoning for covariates which were not zonally averaged
if CoVariateFlat.size > 0:
    for nlon in range(Nlon2):
        for nv in range(NCovarsVec):
            if DetrendCovar == 1:
                CoVariateFlat[nlon, nv, :] = sm.tsa.detrend(CoVariateFlat[nlon, nv, :])
            if DeseasonCovar == 1:
                decomposition = sm.tsa.seasonal_decompose(
                    CoVariateFlat[nlon, nv, :], model="additive", freq=12
                )  # Adjust freq as needed
                seasonal_component = decomposition.seasonal
                CoVariateFlat[nlon, nv, :] = (
                    CoVariateFlat[nlon, nv, :] - seasonal_component
                )

# Detrending and deseasoning for covariates which were zonally averaged
for nv in range(NumInts):
    if DetrendCovar == 1:
        CoVariateScalars[nv, :] = sm.tsa.detrend(CoVariateScalars[nv, :])
    if DeseasonCovar == 1:
        decomposition = sm.tsa.seasonal_decompose(
            CoVariateScalars[nv, :], model="additive", freq=12
        )  # Adjust freq as needed
        seasonal_component = decomposition.seasonal
        CoVariateScalars[nv, :] = CoVariateScalars[nv, :] - seasonal_component

# Normalize data for regression (Global method)
if NetNorm == "Global":
    for nc in range(NCovarsVec):
        CoVariateFlat_0 = CoVariateFlat[
            :, nc, :
        ]  # CoVariateFlat ~ (Nlon, NCovarsVec, Nsamps)
        m = np.min(CoVariateFlat_0)
        M = np.max(CoVariateFlat_0)
        CoVariateFlat_0 = (CoVariateFlat_0 - (M + m) / 2) / ((M - m) / 2)
        CoVariateFlat[:, nc, :] = CoVariateFlat_0

# Reshape as [Nlon2*NCovarsVec, Nsamps]
CoVariateFlat = np.reshape(CoVariateFlat, [Nlon2 * NCovarsVec, Nsamps])
CoVariateFlat = CoVariateFlat.T  # Reshape as (Nsamps, Nlon2*NCovarsVec)

if CoVariateScalars is not None:
    CoVariateFlat = np.hstack(
        (CoVariateFlat, CoVariateScalars)
    )  # Add the zonal-mean variables, if defined.

x = CoVariateFlat  # Rename, since x may undergo filtering.
y = Psi  # Assuming Psi is already a univariate time series
Nlon_pseudo = Nlon2 * NCovarsVec + NumInts

# Define the cutoff frequency and sampling frequency
cutoff_freq = LPF_Freq  # Specify the cutoff frequency for lowpass filter
sampling_freq = 1.0  # Sampling frequency (adjust as needed)
nyquist_freq = 0.5 * sampling_freq  # Calculate the Nyquist frequency
normalized_cutoff_freq = (
    cutoff_freq / nyquist_freq
)  # Calculate the normalized cutoff frequency
filter_order = 4
b, a = butter(
    filter_order, normalized_cutoff_freq, btype="low"
)  # Design the Butterworth lowpass filter

# Apply the lowpass filter to both x and y
x = lfilter(b, a, x, axis=0)
y = lfilter(b, a, y)


def plot(lon, x0, x, bath):
    nt = 10
    fig, ax1 = plt.subplots()

    # Plot x0(nt,:) and x(nt,:) on the left y-axis (ax1)
    color = "tab:blue"
    ax1.set_xlabel("Lon [deg]")
    ax1.set_ylabel("Bottom pressure", color=color)
    ax1.plot(lon, x0[nt, :], label="raw", color=color)
    ax1.plot(lon, x[nt, :], label="smooth", linestyle="--", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.legend(loc="upper left")

    # Create a second y-axis on the right for plotting -bath
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Depth [m]", color=color)
    ax2.plot(lon, -bath, label="bath", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(loc="upper right")

    plt.show()


# Assuming you have DelaySamplesMax, DelaySamplesN, x, CoVariateFlat, y, Psi defined
if np.sum(DelaySamples != 0) > 1:
    if np.all(DelaySamples > 0):
        Nsamps, Nlon_pseudo = x.shape
        DelaySamplesMax = np.max(DelaySamples)

        # Prepare predictors with fewer time samples and more "spatial" samples
        x_d = np.nan * np.zeros(
            (Nsamps - DelaySamplesMax, Nlon_pseudo * (1 + DelaySamplesN))
        )
        CoVariateFlat_d = x_d.copy()
        x_d[:, :Nlon_pseudo] = x[DelaySamplesMax:, :]

        # Populate first parts of padded arrays with undelayed samples
        CoVariateFlat_d[:, :Nlon_pseudo] = CoVariateFlat[DelaySamplesMax:, :]
        y_d = y[DelaySamplesMax:]
        Psi_d = Psi[DelaySamplesMax:]

        # Populate latter parts of padded arrays with delayed samples
        for nd in range(DelaySamplesN):
            x_d[:, (Nlon_pseudo * (nd + 1)) : (Nlon_pseudo * (nd + 2))] = x[
                DelaySamplesMax - DelaySamples[nd] : -DelaySamples[nd], :
            ]
            CoVariateFlat_d[:, (Nlon_pseudo * (nd + 1)) : (Nlon_pseudo * (nd + 2))] = (
                CoVariateFlat[DelaySamplesMax - DelaySamples[nd] : -DelaySamples[nd], :]
            )

        x = x_d
        y = y_d
        CoVariateFlat = CoVariateFlat_d
        Psi = Psi_d

        Nsamps, Nlon_pseudo = x.shape
    else:
        raise ValueError("Not ready for DelaySamples < 0")


# Assuming you have Nsamps and TrainValidTestFracs defined
TrainingSamples = np.arange(
    1, int(np.ceil(Nsamps * TrainValidTestFracs[0]) + 1)
)  # +1 because Python indexing is exclusive
NsampsTrain = len(TrainingSamples)

if len(TrainValidTestFracs) == 3:
    ValidationSamples = np.arange(
        TrainingSamples[-1] + 1,
        TrainingSamples[-1] + int(np.ceil(Nsamps * TrainValidTestFracs[1]) + 1),
    )
    TestSamples = np.arange(ValidationSamples[-1] + 1, Nsamps + 1)
    ValidationAndTestSamples = np.concatenate([ValidationSamples, TestSamples])
elif len(TrainValidTestFracs) == 2:
    ValidationSamples = np.arange(TrainingSamples[-1] + 1, Nsamps + 1)
    TestSamples = ValidationSamples.copy()
    ValidationAndTestSamples = ValidationSamples
else:
    raise ValueError("Incorrect TrainValidTestFracs length")

y1 = y[TrainingSamples - 1]  # Python indexing starts from 0
y2 = y[ValidationSamples - 1]
y3 = y[TestSamples - 1]
y1_raw = Psi[TrainingSamples - 1]
y2_raw = Psi[ValidationSamples - 1]
y3_raw = Psi[TestSamples - 1]

x1 = x[TrainingSamples - 1, :]
x2 = x[ValidationSamples - 1, :]
x3 = x[TestSamples - 1, :]
x1_raw = CoVariateFlat[TrainingSamples - 1, :]
x2_raw = CoVariateFlat[ValidationSamples - 1, :]
x3_raw = CoVariateFlat[TestSamples - 1, :]

Years = np.arange(1, Nsamps + 1) / 12

# LINEAR REGRESSION
if LinReg == 0:
    IncludeIntercept = True
    if IncludeIntercept:
        x_r = np.hstack((x, np.ones((Nsamps, 1))))
        x1_r = np.hstack((x1, np.ones((len(TrainingSamples), 1))))
        x2_r = np.hstack((x2, np.ones((len(ValidationSamples), 1))))
    else:
        x_r = x
        x1_r = x1
        x2_r = x2

    model = LinearRegression()
    model.fit(x1_r, y1)
    y1p = model.predict(x1_r)
    y2p = model.predict(x2_r)

    cc = np.corrcoef(y2, y2p[ValidationAndTestSamples - 1])[0, 1]

    plt.figure()
    plt.plot(Years, y, label="Psi")
    plt.plot(Years, y1p, label="Linear Regression")
    plt.legend()
    plt.title(f"Linear Regression. Valid corr={cc}")
    figfn = f"{OutputFolder}Regress_Moc{PsiStr}And{CoVariateNamesStr}{FnFiltTime}{FnFiltLon}_LinReg.png"
    plt.savefig(figfn, dpi=300)
    plt.close()

    MatFN = f"{OutputFolder}Regress_Moc{PsiStr}And{CoVariateNamesStr}{FnFiltTime}{FnFiltLon}_LinReg.mat"
    np.savez(
        MatFN,
        TrainingSamples=TrainingSamples,
        ValidationSamples=ValidationSamples,
        CoVariateNamesStr=CoVariateNamesStr,
        cc=cc,
        PsiMethod=PsiMethod,
        latrange=latrange,
        lat0=lat0,
        FiltType=FiltType,
        FiltWind=FiltWind,
        lat1=lat1,
        lat2=lat2,
        ZonalSmoothRadDeg=ZonalSmoothRadDeg,
        LPF_Freq=LPF_Freq,
        LonsNotNan=LonsNotNan,
        IndsNotNanX=IndsNotNanX,
        LonDec=LonDec,
        y1=y1,
        y2=y2,
        y1p=y1p,
        y2p=y2p,
        y=y,
        yp=yp,
        EccoYear0=EccoYear0,
        Years=Years,
    )

# LINEAR REGRESSION OF GRADIENT
if LinRegGrad == 1:
    ZonalBinning = True
    lambda_values = [1, 1e2, 1e3, 1e4, 1e5, 1e6]

    for CG in [1]:  # Change the list of CG values as needed
        if ZonalBinning:
            x1_b = np.mean(np.reshape(x1, (NsampsTrain, Nlon2 // CG, CG)), axis=2)
            x1_b2 = np.hstack(
                (x1_b[:, -1][:, np.newaxis], x1_b, x1_b[:, 0][:, np.newaxis])
            )
            x1_b = 0.5 * (x1_b2[:, 2:] - x1_b2[:, :-2])
            x1_b = np.hstack((x1_b, np.ones((NsampsTrain, 1))))

            model = LassoCV(alphas=lambda_values, cv=10)
            model.fit(x1_b, y1)
            y1p_lasso = model.predict(x1_b)

            x_b = np.mean(np.reshape(x, (Nsamps, Nlon2 // CG, CG)), axis=2)
            x_b2 = np.hstack((x_b[:, -1][:, np.newaxis], x_b, x_b[:, 0][:, np.newaxis]))
            x_b = 0.5 * (x_b2[:, 2:] - x_b2[:, :-2])
            x_b = np.hstack((x_b, np.ones((Nsamps, 1))))
            y2p_lasso = model.predict(x_b)
        else:
            model = LassoCV(alphas=lambda_values, cv=10)
            model.fit(x1, y1)
            y1p_lasso = model.predict(x1)
            y2p_lasso = model.predict(x)

        cc_lasso = np.corrcoef(
            Psi[ValidationAndTestSamples], np.hstack((y2p_lasso, y3p_lasso))
        )[0, 1]

        plt.figure()
        plt.plot(Years, y, label="Psi")
        plt.plot(Years, y1p_lasso, label="Lasso Linear Regression")
        plt.legend()
        plt.title(
            f"LASSO Linear Regression of zonal-grad. CG={CG}. Lasso-Valid corr={cc_lasso}"
        )
        figfn = f"{OutputFolder}Regress_Moc{PsiStr}And{CoVariateNamesStr}{FnFiltTime}{FnFiltLon}_LASSO_CG{CG}.png"
        plt.savefig(figfn, dpi=300)
        plt.close()


# ROBUST LINEAR REGRESSION
if LinRegRobust == 1:
    CG = 4
    x1_degraded = np.mean(np.reshape(x1, (len(y1), Nlon2 // CG, CG)), axis=2)

    huber_reg = HuberRegressor().fit(x1_degraded, y1)
    y1p_robust = huber_reg.predict(x1_degraded)

    b2 = np.convolve(huber_reg.coef_, np.ones(CG) / CG, mode="valid")
    Psi_fit_robust = huber_reg.intercept_ + np.sum(b2 * x, axis=1)

    plt.figure()
    plt.plot(Years, y, label="Psi")
    plt.plot(Years, Psi_fit_robust, label="Robust Linear Regression")
    plt.legend()
    plt.title(f"Robust Linear Regression of zonal-grad. CG={CG}")
    figfn = f"{OutputFolder}Regress_Moc{PsiStr}And{CoVariateNamesStr}{FnFiltTime}{FnFiltLon}_Robust_CG{CG}.png"
    plt.savefig(figfn, dpi=300)
    plt.close()

# Perform stepswise regression
x1 = sm.add_constant(x1)
mdl = sm.OLS(y1, x1).fit()

# Initialize lists to store results
R2_training = []
mse_training = []
corrcoef_training = []

# Loop through different numbers of neurons in the hidden layers
for Nneurons in NneuronsList:
    for nrep in range(NNrepeats):
        model = Sequential()  # Feedforward neural network model
        model.add(
            Dense(units=Nneurons, activation="relu", input_dim=x.shape[1])
        )  # Add input layer
        model.add(Dense(units=Nneurons, activation="relu"))  # Add hidden layers
        model.add(Dense(units=Nneurons, activation="relu"))
        model.add(Dense(units=1, activation="linear"))  # Add output layer

        model.compile(loss="mean_squared_error", optimizer="adam")  # Compile the model
        history = model.fit(  # Train the model
            x[TrainingSamples],
            y[TrainingSamples],
            epochs=100,
            batch_size=32,
            validation_data=(x[ValidationSamples], y[ValidationSamples]),
            verbose=0,
        )

        # Evaluate the model on training data
        y_pred_training = model.predict(x[TrainingSamples])
        R2_training.append(calculate_R2(y[TrainingSamples], y_pred_training))
        mse_training.append(calculate_mse(y[TrainingSamples], y_pred_training))
        corrcoef_training.append(
            np.corrcoef(y[TrainingSamples], y_pred_training.flatten())[0, 1]
        )

        y1p = model.predict(x1)  # Predictions for the training set
        y2p = model.predict(x2)  # Predictions for the validation set
        y3p = model.predict(x3)  # Predictions for the testing set
        yp = model.predict(x)  # Predictions for the entire dataset

        # Calculate correlation coefficient (cc)
        cc_training = np.corrcoef(y1, y1p.flatten())[0, 1]
        cc_validation = np.corrcoef(y2, y2p.flatten())[0, 1]
        cc_testing = np.corrcoef(y3, y3p.flatten())[0, 1]
        cc_complete = np.corrcoef(y, yp.flatten())[0, 1]

        # Calculate R-squared (R2)
        def calculate_R2(y_true, y_pred):
            SS_res = np.sum((y_true - y_pred) ** 2)
            SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (SS_res / SS_tot)

        R2_training = calculate_R2(y1, y1p.flatten())
        R2_validation = calculate_R2(y2, y2p.flatten())
        R2_testing = calculate_R2(y3, y3p.flatten())
        R2_complete = calculate_R2(y, yp.flatten())

        # Calculate mean squared error (mse)
        mse_training = np.mean((y1 - y1p.flatten()) ** 2)
        mse_validation = np.mean((y2 - y2p.flatten()) ** 2)
        mse_testing = np.mean((y3 - y3p.flatten()) ** 2)
        mse_complete = np.mean((y - yp.flatten()) ** 2)

        # Print the results
        print(
            f"Training correlation: {cc_training}, R-squared: {R2_training}, MSE: {mse_training}"
        )
        print(
            f"Validation correlation: {cc_validation}, R-squared: {R2_validation}, MSE: {mse_validation}"
        )
        print(
            f"Testing correlation: {cc_testing}, R-squared: {R2_testing}, MSE: {mse_testing}"
        )
        print(
            f"Complete dataset correlation: {cc_complete}, R-squared: {R2_complete}, MSE: {mse_complete}"
        )

        """
        from scipy.stats import pearsonr
        corrcoef_training_0 = pearsonr(y1, y1t)
        corrcoef_training[NneurN, nrep] = corrcoef_training_0[
            0
        ]  # Use the first value, which is the correlation coefficient

        corrcoef_validation_0 = pearsonr(Psi[ValidationSamples], y2_raw)
        corrcoef_validation[NneurN, nrep] = corrcoef_validation_0[
            0
        ]  # Use the first value, which is the correlation coefficient
        """

        if NNrepeats > 1:
            nrepstr = "_rep" + str(nrep)
        else:
            nrepstr = ""

        # Plotting Psi and its fit
        plt.figure()
        plt.plot(EccoYear0 + Years, y / 1e6, label="Psi")
        plt.plot(EccoYear0 + Years, yp / 1e6, label="Fit")
        plt.xlim([EccoYear0 + Years[0], EccoYear0 + Years[-1]])
        plt.xlabel("Time [years]")
        plt.ylabel("Transport anomaly [Sv]")
        plt.legend()
        plt.title(
            [
                BasinStrTitle,
                BotttomMOCstrTitle,
                PsiStr + " vs " + CoVariateNamesStr,
                TitleFiltTime[2:],
                TitleFiltLon,
                "Neural network fit. #neurons=" + StrNneurons + ". Repeat#" + str(nrep),
                "Validation r^2="
                + str(corrcoef_validation[NneurN, nrep] ** 2)
                + ". Testing r^2="
                + str(corrcoef_testing[NneurN, nrep] ** 2),
                "Validation mse="
                + str(mse_validation[NneurN, nrep])
                + ". Testing mse="
                + str(mse_testing[NneurN, nrep])
                + " Sv",
                "Validation R^2="
                + str(R2_validation[NneurN, nrep])
                + ". Testing R^2="
                + str(R2_testing[NneurN, nrep]),
            ]
        )
        figfn = (
            OutputFolder
            + "Regress_Moc"
            + PsiStr
            + "And"
            + CoVariateNamesStr
            + FnFiltTime[2:]
            + FnFiltLon
            + "_NN"
            + StrNneurons
            + nrepstr
            + ".png"
        )
        plt.savefig(figfn, dpi=300)
        plt.close()

        if ActivFunc == "purelin" and len(CoVariateNames) == 1:
            IW = net.IW[0]

            if len(IW) < len(lon):
                IW2 = np.full(len(lon), np.nan)
                for n in range(Nlon2):
                    IW2[IndsNotNanX[n]] = IW[n]
                IW = IW2

            fig, ax1 = plt.subplots(figsize=(10, 4))

            color = "tab:red"
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Weights", color=color)
            ax1.plot(lon, IW, color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = "tab:blue"
            ax2.set_ylabel(
                "Bathymetry [m]", color=color
            )  # we already handled the x-label with ax1
            ax2.plot(lon, -bath, color=color)
            ax2.tick_params(axis="y", labelcolor=color)

            plt.title(
                [
                    BasinStrTitle,
                    BotttomMOCstrTitle,
                    PsiStr + " vs " + CoVariateNamesStr,
                    TitleFiltTime[2:],
                    TitleFiltLon,
                    "Neural network fit. #neurons="
                    + StrNneurons
                    + ". Repeat#"
                    + str(nrep),
                    "Linear NN weights (left) and bathymetry (right)",
                ]
            )
            figfn = (
                OutputFolder
                + "Regress_Moc"
                + PsiStr
                + "And"
                + CoVariateNamesStr
                + FnFiltTime[2:]
                + FnFiltLon
                + "_NN"
                + StrNneurons
                + nrepstr
                + "_WeightsVsBath.png"
            )
            plt.savefig(figfn, dpi=300)
            plt.close()

        """
        -- Plot raw data, linear fit and NN fit in time -- 
        
        b1 = [0, 0.4470, 0.7410]
        r1 = [0.8500, 0.3250, 0.0980]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(EccoYear0 + Years, y / 1e6, color=b1, label='Raw data')
        ax.plot(EccoYear0 + Years, Psi_fit_Lin / 1e6, '--', color='gray', label='Linear Fit')
        ax.plot(EccoYear0 + Years, yp / 1e6, color=r1, label='NN fit')
        ax.axvline(EccoYear0 + (Years[-1] - Years[0]) * TrainValidTestFracs[0], color='black')
        ax.axvline(EccoYear0 + (Years[-1] - Years[0]) * sum(TrainValidTestFracs[:2]), color='black')

        plt.xlim([EccoYear0 + Years[0], EccoYear0 + Years[-1]])
        plt.ylim([-15, 15])
        plt.grid()
        plt.xlabel('Time [years]')
        plt.ylabel('Transport [10^6 m^3/s]')
        plt.title('Southern Ocean Deep Meridional Overturning Circulation')
        plt.legend(loc='southwest')

        figfn = OutputFolder + 'Regress_Moc' + PsiStr + 'And' + CoVariateNamesStr + FnFiltTime[
                                                                                    2:] + FnFiltLon + '_NN' + StrNneurons + nrepstr + '.png'
        plt.savefig(figfn, dpi=300)
        plt.close()
        """

        """
        -- Plot the correlation between CoVariate and PSI -- 
        
        # Define the colormap and color levels
        cmap = 'viridis'
        rmax = np.max(np.abs(b_reshaped))
        cc = np.linspace(-rmax, rmax, 20)

        # Create the filled contour plot
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(Lon, Lat, b_reshaped, levels=cc, cmap=cmap)
        plt.colorbar(contour, ax=ax)

        # Add labels and title
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Correlation between CoVariate and PSI')

        plt.show()
        """

        data = {"net": net, "YearsECCO": YearsECCO, "Psi": Psi}

        file_path = "NN20_unfilt_smoothrad2.mat"
        savemat(file_path, data)

        """
        -- SAVING ALL THE PARAMETERS -- 
        
        # Define the variables you want to save
        variables = {
            'net': net,
            'Nneurons': Nneurons,
            'TrainingSamples': TrainingSamples,
            'ValidationSamples': ValidationSamples,
            'CoVariateNamesStr': CoVariateNamesStr,
            'corrcoef_training': corrcoef_training,
            'corrcoef_validation': corrcoef_validation,
            'corrcoef_testing': corrcoef_testing,
            'PsiMethod': PsiMethod,
            'latrange': latrange,
            'lat0': lat0,
            'FiltType': FiltType,
            'FiltWind': FiltWind,
            'lat1': lat1,
            'lat2': lat2,
            'ZonalSmoothRadDeg': ZonalSmoothRadDeg,
            'FiltType': FiltType,
            'LPF_Freq': LPF_Freq,
            'LonsNotNan': LonsNotNan,
            'IndsNotNanX': IndsNotNanX,
            'LonDec': LonDec,
            'y1': y1,
            'y2': y2,
            'y3': y3,
            'y1p': y1p,
            'y2p': y2p,
            'y3p': y3p,
            'y': y,
            'yp': yp,
            'EccoYear0': EccoYear0,
            'Years': Years
        }

        file_path = MatFN
        savemat(file_path, variables)
        """

    if NNrepeats > 1:
        additional_variables = {
            "ECCOv": ECCOv,
            "NNrepeats": NNrepeats,
            "NneuronsList": NneuronsList,
            "R2_training": R2_training,
            "R2_validation": R2_validation,
            "R2_testing": R2_testing,
            "mse_training": mse_training,
            "mse_validation": mse_validation,
            "mse_testing": mse_testing,
            "LonDec": LonDec,
        }
        variables.update(additional_variables)

        file_path = [
            OutputFolder,
            CoVariateNamesStr,
            "VsMoc",
            PsiStr,
            FnFiltTime,
            FnFiltLon,
            FnCorrDesc,
            "_NN",
            StrNneurons,
            "_",
            "_",
            trainFcn,
            "_RepeatMatrix",
            str(NNrepeats),
            ".mat",
        ]
        sio.savemat(file_path, variables)

if len(NneuronsList) * len(NNrepeats) > 1:

    NneuronsMat, RepsMat = np.meshgrid(NneuronsList, np.arange(1, NNrepeats + 1))

    # Create a figure for the validation set correlation matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corrcoef_validation, cmap="viridis")

    # Customize the colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.label.set_text("Correlation")

    # Set ticks and labels for y-axis
    yticks = np.arange(len(NneuronsList))
    ax.set_yticks(yticks)
    ax.set_yticklabels(NneuronsList)

    # Set ticks and labels for x-axis
    xticks = np.arange(NNrepeats)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks + 1)  # Adding 1 to start the count from 1

    # Add labels and title
    ax.set_xlabel("Repeat #")
    ax.set_ylabel("# Neurons")
    ax.set_title(
        f"Neural network fits: validation set correlation\n{BasinStrTitle}, {BotttomMOCstrTitle}, {PsiStr} vs {CoVariateNamesStr}, {TitleFiltTime[2:]}, {TitleFiltLon}"
    )

    # Save the figure
    figfn = f"{OutputFolder}{CoVariateNamesStr}VsMoc{PsiStr}{FnFiltTime}{FnFiltLon}{FnCorrDesc}_NN_RepeatMatrix{NNrepeats}_MatValidation.png"
    plt.savefig(figfn, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a figure for the testing set correlation matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(corrcoef_testing, cmap="viridis")

    # Customize the colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.label.set_text("Correlation")

    # Set ticks and labels for y-axis
    yticks = np.arange(len(NneuronsList))
    ax.set_yticks(yticks)
    ax.set_yticklabels(NneuronsList)

    # Set ticks and labels for x-axis
    xticks = np.arange(NNrepeats)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks + 1)  # Adding 1 to start the count from 1

    # Add labels and title
    ax.set_xlabel("Repeat #")
    ax.set_ylabel("# Neurons")
    ax.set_title(
        f"Neural network fits: testing set correlation\n{BasinStrTitle}, {BotttomMOCstrTitle}, {PsiStr} vs {CoVariateNamesStr}, {TitleFiltTime[2:]}, {TitleFiltLon}"
    )

    # Save the figure
    figfn = f"{OutputFolder}{CoVariateNamesStr}VsMoc{PsiStr}{FnFiltTime}{FnFiltLon}{FnCorrDesc}_NN_RepeatMatrix{NNrepeats}_MatTesting.png"
    plt.savefig(figfn, dpi=300, bbox_inches="tight")
    plt.close()

    ## Find a good minimal ylim value for both the validation and testing
    Rmin4plot = min(np.min(mean_corrcoef_validation), np.min(mean_corrcoef_testing))
    Rmin4plot = np.floor(Rmin4plot * 10) / 10

    # Create the figure
    fh = plt.figure()
    plt.errorbar(
        NneuronsList,
        mean_corrcoef_validation,
        std_corrcoef_validation,
        linewidth=1,
        label="Validation",
    )
    plt.errorbar(
        NneuronsList,
        mean_corrcoef_testing,
        std_corrcoef_testing,
        linewidth=1,
        label="Testing",
    )
    plt.legend(loc="best")
    plt.xlabel("# neurons")
    plt.ylabel("Correlation")
    plt.grid()
    plt.ylim([Rmin4plot, 1])
    plt.title(
        [
            "BasinStrTitle",
            "BotttomMOCstrTitle",
            "PsiStr vs CoVariateNamesStr",
            "TitleFiltTime",
            "TitleFiltLon",
            "Neural network fits #repeats=3, correlation",
        ]
    )

    # Save the figure
    figfn = (
        "OutputFolder" + "TestingValidationCor.png"
    )  # Replace with your actual output folder
    plt.savefig(figfn, dpi=300, bbox_inches="tight")
    plt.show()

    # Create the MSE plot
    plt.figure()
    plt.errorbar(
        NneuronsList,
        np.mean(mse_validation, axis=1),
        np.std(mse_validation, axis=1),
        linewidth=1,
    )
    plt.errorbar(
        NneuronsList,
        np.mean(mse_testing, axis=1),
        np.std(mse_testing, axis=1),
        linewidth=1,
    )
    plt.legend(["Validation", "Testing"], loc="best")
    plt.xlabel("# neurons")
    plt.ylabel("MSE [Sv]")
    plt.grid()
    plt.title(
        [
            BasinStrTitle,
            BotttomMOCstrTitle,
            PsiStr,
            " vs ",
            CoVariateNamesStr,
            TitleFiltTime[2:],
            TitleFiltLon,
            "Neural network fits #repeats=" + str(NNrepeats) + ", MSE",
        ]
    )
    figfn = (
        OutputFolder
        + CoVariateNamesStr
        + "VsMoc"
        + PsiStr
        + FnFiltTime[2:]
        + FnFiltLon
        + FnCorrDesc
        + "NN_RepeatMatrix"
        + str(NNrepeats)
        + "_TestingValidationMSE.png"
    )
    plt.savefig(figfn, dpi=300, bbox_inches="tight")

    # Create the R^2 plot
    plt.figure()
    plt.errorbar(
        NneuronsList,
        np.mean(R2_validation, axis=1),
        np.std(R2_validation, axis=1),
        linewidth=1,
    )
    plt.errorbar(
        NneuronsList,
        np.mean(R2_testing, axis=1),
        np.std(R2_testing, axis=1),
        linewidth=1,
    )
    plt.legend(["Validation", "Testing"], loc="best")
    plt.xlabel("# neurons")
    plt.ylabel("R^2 [-]")
    plt.grid()
    plt.title(
        [
            BasinStrTitle,
            BotttomMOCstrTitle,
            PsiStr,
            " vs ",
            CoVariateNamesStr,
            TitleFiltTime[2:],
            TitleFiltLon,
            "Neural network fits #repeats="
            + str(NNrepeats)
            + ", coefficient of determination",
        ]
    )
    figfn = (
        OutputFolder
        + CoVariateNamesStr
        + "VsMoc"
        + PsiStr
        + FnFiltTime[2:]
        + FnFiltLon
        + FnCorrDesc
        + "NN_RepeatMatrix"
        + str(NNrepeats)
        + "_TestingValidationR2.png"
    )
    plt.savefig(figfn, dpi=300, bbox_inches="tight")

"""
IW11 = net.IW[0]  # Assuming IW11 is a 2D array

nneur = 1  # Change this to the desired neuron index
plt.figure()
plt.plot(lon, IW11[nneur, :])
plt.twinx()
plt.plot(lon, -bath)

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(IW11[nneur, :], bath)[0, 1]
print(f"Correlation Coefficient: {correlation_coefficient}")

plt.show()
"""

import numpy as np

"""
-- Aviv 2021-02-15 --
Pull out a zonal strip from ECCOV4r3 data. Works northward of -70N, and south of (?) 50N, i.e., where the grid is lon-lat.

-- Aviv 2022-03-12 --
 Correct the SizeMin-SizeMax allocations to lon-lat, i.e., 
 if "lats">90, the previous (V2) allocation of lats=SizeMin is wrong (as #lons=90)
"""

lat1 = -70
lat2 = -50


def LLC2ZonalStrip_V3(mygrid, Var, lat1, lat2):

    nfaces = [1, 2, 4, 5]

    LatA = mygrid["YC"]

    # If LatA is a NumPy array or an xarray DataArray, you can check its min and max
    if hasattr(LatA, 'min') and hasattr(LatA, 'max'):
        print(
            f"Latitude range: {LatA.min().values if hasattr(LatA.min(), 'values') else LatA.min()} to {LatA.max().values if hasattr(LatA.max(), 'values') else LatA.max()}")

    # Find indices within latitude bounds
    Indices = np.where((LatA >= lat1) & (LatA <= lat2))
    print(Indices)
    Ind1 = Indices[0]
    Ind2 = Indices[1]

    Size1 = np.max(Ind1) - np.min(Ind1) + 1
    Size2 = np.max(Ind2) - np.min(Ind2) + 1
    SizeMin, SizeMax = np.min([Size1, Size2]), np.max([Size1, Size2])

    if SizeMin == 90:
        VarNew = np.nan * np.zeros(
            (SizeMax, SizeMin * 4)
        )  # Assuming SizeMin==90 implies longitude dimension
    else:
        VarNew = np.nan * np.zeros(
            (SizeMin, SizeMax * 4)
        )  # Assuming SizeMax==90 implies longitude dimension

    _, Ncols = VarNew.shape
    LonNew, LatNew = np.copy(VarNew), np.copy(VarNew)

    col1, col2 = 0, Ncols // 4

    # Adjust based on actual data structure
    for n in nfaces:
        LonA = mygrid["XC"][n]
        LatA = mygrid["YC"][n]
        VarFace = Var[n]

        # Find indices again, adjusting bounds slightly
        Indices = np.where((LatA >= lat1) & (LatA <= lat2))
        Ind1 = Indices[0]
        Ind2 = Indices[1]
        print(Ind1)
        print(Ind2)
        Ind1 = np.arange(np.min(Ind1), np.max(Ind1) + 1)
        Ind2 = np.arange(np.min(Ind2), np.max(Ind2) + 1)

        LonA = LonA[np.ix_(Ind1, Ind2)]
        LatA = LatA[np.ix_(Ind1, Ind2)]
        VarFace = VarFace[np.ix_(Ind1, Ind2)]

        if len(Ind1) == 90:
            LonA, LatA, VarFace = LonA.T, LatA.T, VarFace.T

        if LonA[0, 0] > LonA[0, 1]:
            VarFace = np.fliplr(VarFace)
            LonA = np.fliplr(LonA)
            LatA = np.fliplr(LatA)

        a, _ = LatA.shape
        if a > 1 and LatA[0, 0] > LatA[1, 0]:
            VarFace = np.flipud(VarFace)
            LonA = np.flipud(LonA)
            LatA = np.flipud(LatA)

        VarNew[:, col1:col2] = VarFace
        LonNew[:, col1:col2] = LonA
        LatNew[:, col1:col2] = LatA

        col1 = col2
        col2 = col1 + Ncols // 4

    # Sort by longitude to move the 180-degree line
    IndSort = np.argsort(LonNew[0, :])
    LonNew, LatNew, VarNew = LonNew[:, IndSort], LatNew[:, IndSort], VarNew[:, IndSort]
    return LonNew, LatNew, VarNew

import os
import matplotlib.pyplot as plt

from pathlib import Path

import utils

# Define initial conditions
OS = 'Linux'  # or 'Windows'
ECCOv = 'V4r3'  # or 'V4r4'

HOMEDIR_3 = "C:/Users/aline/OTP/ECCO4_Release3/"
HOMEDIR_4 = "C:/Users/aline/OTP/ECCO4_Release4/"

# Define base directories based on ECCO version
if ECCOv == 'V4r4':
    dirv4r4 = Path(HOMEDIR_4)
else:
    dirv4r3 = Path(HOMEDIR_3)

# Add paths for MATLAB scripts (for Python, you would ensure your modules are in the PYTHONPATH or use equivalent Python packages)
p = Path('D:/Aviv/Research/MATLAB_Scripts/Ocean/gcmfaces/').resolve()

# Set dirGrid and OutputFolder based on ECCO version
if ECCOv == 'V4r3':
    dirGrid = Path(os.path.join(HOMEDIR_3+'nctiles_grid')).resolve()
    OutputFolder = dirv4r3
else:
    # Assuming this is meant to be set only for 'V4r4'
    OutputFolder = Path(HOMEDIR_4).resolve()

mygrid = utils.grid_load(dirGrid)
#grid = utils.convert2pcol(mygrid.XC, mygrid.YC, mygrid.Depth)

plt.figure()
plt.pcolormesh(mygrid.XC, mygrid.YC, mygrid.Depth, shading='flat', cmap='viridis')
plt.colorbar()
plt.xlim([-180, 180])
plt.show()

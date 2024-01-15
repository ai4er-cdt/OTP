# Setup

1. Clone this repository using `git clone --recursive <HTTPS or SSH>` to ensure the ECCO submodule is loaded as well.
2. Edit the `environment.yml` file to include your own local filepath for your conda environments. See comment in file for further instructions.
3. Create your conda python environment using `conda env create -f environment.yml` and activate the environment with `conda activate gtc`.
4. Make sure you can run the first cell of `sandbox.ipynb`--this ensures that the ECCO package is loaded correctly and all dependencies have been installed to your python environment. If using VSCode, be sure to set your kernel to the `gtc` conda environment.
# Repository Documentation

This file contains a high-level overview of the repository structure and the role of each file within. For more details about any individual file or function, please consult the file directly--they are all relatively well commented. Notebooks include additional explanation as well.

_If you find any bugs in the code or have trouble setting up your environment, please open an issue or contact us directly! Our emails are linked in the README and we'd be happy to help._

## Environment Setup

To run the project locally, we recommend you set up your environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

1. Clone this repository using `git clone --recursive <HTTPS or SSH>` to ensure the ECCO submodule is loaded as well.
2. Navigate to the newly created repository directory (i.e., use `cd OTP`).
3. Edit the `environment.yml` file to include your own local filepath for your `conda` environments.
   - If you're not sure what this filepath is, you can check using `conda info --envs`. The filepath should be `[FILEPATH FOR BASE ENVIRONMENT]/envs/`. Notice that this ends in `gtc`--this will be the name of the created `conda` environment.
4. Create your `conda` Python environment using `conda env create -f environment.yml`.
5. Activate the environment with `conda activate gtc`.
6. Make sure you can run the first code cell of `setup_verification.ipynb` in the main project directory--this ensures that the ECCO submodule has been downloaded correctly and that all dependencies have been installed to your Python environment.
   - `jupyter-lab` will be installed in the newly created environment, so you can run this notebook by spinning up Jupyter Lab (input the `jupyter-lab` command on the command line while in the project directory) and opening the notebook. Be sure you activate the `gtc` environment before opening Jupyter Lab.
   - If using VSCode to run this notebook, be sure to set your kernel to the `gtc` conda environment.

## Repository Overview

Many directories containn an `archive/` sub-directory, which contains code that was produced during the project, but is not needed to reproduce our final analysis. Many of these archived files are well-commented or self-explanatory, but given their secondary nature we do not describe them in the same level of detail in this overview.

### `scripts/`

This directory contains all Python scripts used for data download and preprocessing.

- `basin_masks.py`: Implements functionality for extracting basin masks to match \[1\] (Southern, Atlantic, Indo-Pacific).
- `download_ecco.py`: Wrapper script to download all ECCO data needed for analyses (surface variables and velocity fields).
- `ecco_download.py`: Download functionality to interface with NASA PO.DAAC data repository, where ECCO is stored. This is borrowed from the [ECCO Python package tutorial](https://github.com/ECCO-GROUP/ECCO-v4-Python-Tutorial).
- `streamfunction_latlon.py`: Custom functionality for calculating the overturning streamfunction at a latitude in both depth- and density-space.

### `models/`

This directory contains all machine learning model definitions (mostly different forms of neural networks), `PyTorch` dataset class definitions, model training loops, and utility functions for use during modelling.

**Model definitions:**
- `MLP.py`: A basic multi-layer perceptron architecture, with options for multiple hidden layers with variable numbers of neurons and dropout.
- `SOLODOCH.py`: A precise replication of the neural network architecture used in \[1\]. Otherwise the same as `MLP.py`.
- `CNN1D.py`: A 1-dimensional convolutional neural network, with both "pure" layers (i.e., independent filters for each feature) and "mix" layers (i.e., filters that act on multiple features at once). Options are also included for dropout, number of layers, and number of filters. This model is usually used to convolve over longitudes when input variables contain full zonal information.
- `CNN2D.py`: A 2-dimensional convolutional neural network, with both "pure" layers and "mix" layers. Options are also included for dropout, number of layers, and number of filters. This model is usually used to convolve over a latitudinal strip and over longitudes.
- `CNN3D.py`: A 3-dimensional convolutional neural network, with both "pure" layers and "mix" layers. Options are also included for dropout, number of layers, and number of filters. We didn't use this model in our analysis, but its intended use it for convolving over latitudes, longitudes, and through time.
- `CNN_RAPID.py`: A custom 3-dimensional convolutional neural network that adds an encoding for RAPID data which is concatenated after the convolutions are applied to surface variables. Otherwise the same as `CNN2D.py`
- `ESN.py`: An implementation of an Echo State Network, which is a fully-autoregressive deep learning model used in dynamical systems theory. We didn't use this model in our analysis, but its intended use was to help in predicting circulatory tipping points.

**PyTorch dataset definitions:**
- `SimDataset.py`: A minimal dataset wrapper for use with a `PyTorch` dataloader.
- `RAPIDDataset.py`: An extension of `SimDataset.py` that accounts for the auxiliary RAPID input.

**Utility functions:**
- `utils.py`: Many utility functions to facilitate modelling--data preprocessing, calculating metrics, and saving results.
- `plotting_utils.py`: Two helpful plots for visually assessing model performance in reconstructing the MOC strength.

**Model training loops:**
- `train.py`: A common training loop to be used for all neural network models. A mean squared error loss is used and AdamW is used for optimizing the model weights. Functionality is also provided for saving the model weights and training curve.
- `train_alt.py`: A slight extension of `train.py` that includes early stopping and accomodates RAPID inputs by using `RAPIDDataset.py`.

### `notebooks/`

This directory contains all of our major data processing and all modelling experiments for ECCO.

**Data processing & exploration:**
- `streamfunction/`:
- `moc/`:

**Linear regression experiments:**
- `linear_regression.ipynb`:
- `latitude_transfer_linear_regression`:

**CNN experiments:**
- `neural_networks.ipynb`:

**RAPID experiments:**
- `RAPID_transfer_linear_regression.ipynb`:
- `RAPID_transfer_neural_network.ipynb`:

### `ACCESS/`

This directory contains all major data processing and all modelling experiments for ACCESS. Much of this code has been adpated from code found elsewhere in the repository, and so has been left in its own directory since it is a substantial extension.

**Found elsewhere in repo:**
- `MLP.py`: A basic multi-layer perceptron architecture, with options for multiple hidden layers with variable numbers of neurons and dropout.
- `ESN.py`: An implementation of an Echo State Network, which is a fully-autoregressive deep learning model used in dynamical systems theory. We didn't use this model in our analysis, but its intended use was to help in predicting circulatory tipping points.
- `SimDataset.py`: A minimal dataset wrapper for use with a `PyTorch` dataloader.
- `plotting_utils.py`: Contains functions required to generate timeseries comparison plots as shown in the final report, as well as scatter plots to visualise the performance of regression techniques.
- `utils.py`: Provides preprocessing functions that are customised to act on ACCESS data, including code to remove the trend and seasonality using an additive model.
- `train.py`: A common training loop to be used for all neural network models. A mean squared error loss is used and AdamW is used for optimizing the model weights. Functionality is also provided for saving the model weights and training curve.

**Unique to ACCESS:**
- `data_retrieval.ipynb`: Used to retrieve ACCESS data from the `data/` folder, preprocess input variables, and calculate the MOC. These Xarray dataarrays are then saved to the `processed_data/` folder.
- `models.ipynb`: Used to construct and fit machine learning models in order to produce the results presented in Section **blah** of our final report. Models include (regularised) linear regression, multi-layer perceptrons, echo state networks, gaussaian process regression, and XGBoost. Data required to run this notebook is produced by the `data_retrieval.ipynb` notebook.

# References
[1] Solodoch, A., Stewart, A. L., McC. Hogg, A., & Manucharyan, G. E. (2023). Machine Learning‐Derived Inference of the Meridional Overturning Circulation From Satellite‐Observable Variables in an Ocean State Estimate. _Journal of Advances in Modeling Earth Systems_, 15(4), e2022MS003370.

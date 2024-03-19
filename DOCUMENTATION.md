# Repository Documentation

This file contains a high-level overview of the repository structure and the role of each file within. For more details about any individual file or function, please consult the file directly--they are all relatively well commented. Notebooks include additional explanation as well.

_If you find any bugs in the code or have trouble setting up your environment, please open an issue or contact us directly! Our emails are linked in the README and we'd be happy to help._

-----

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

-----

## Repository Overview

Many directories containn an `archive/` sub-directory, which contains code that was produced during the project, but is not needed to reproduce our final analysis. Many of these archived files are well-commented or self-explanatory, but given their secondary nature we do not describe them with the same level of detail in this overview.

-----

### `scripts/`

This directory contains all Python scripts used for data download and preprocessing.

- `basin_masks.py`: Implements functionality for extracting basin masks to match \[1\] (Southern, Atlantic, Indo-Pacific).
- `ecco_download.py`: Download functionality to interface with NASA PO.DAAC data repository, where ECCO is stored. This is borrowed from the [ECCO Python package tutorial](https://github.com/ECCO-GROUP/ECCO-v4-Python-Tutorial).

-----

### `models/`

This directory contains all machine learning model definitions (mostly different forms of neural networks), `PyTorch` dataset class definitions, model training loops, and utility functions for use during modelling.

**Model definitions:**
- **Fully-connected networks:**
   - `MLP.py`: A basic multi-layer perceptron architecture, with options for multiple hidden layers with variable numbers of neurons and dropout.
   - `SOLODOCH.py`: A precise replication of the neural network architecture used in \[1\]. Otherwise the same as `MLP.py`.
   - `LINMAP.py`: A very basic linear mapping between inputs and outputs--esentially regularised linear regression, but implemented as a `PyTorch` model to take advantage of the computationally efficient stochastic gradient descent in the case of full Southern Ocean training.
- **Convolutional neural networks:**
   - `CNN1D.py`: A 1-dimensional convolutional neural network, with both "pure" layers (i.e., independent filters for each feature) and "mix" layers (i.e., filters that act on multiple features at once). Options are also included for dropout, number of layers, and number of filters. This model is usually used to convolve over longitudes when input variables contain full zonal information.
   - `CNN2D.py`: A 2-dimensional convolutional neural network, with both "pure" layers and "mix" layers. Options are also included for dropout, number of layers, and number of filters. This model is usually used to convolve over a latitudinal strip and over longitudes.
   - `CNN3D.py`: A 3-dimensional convolutional neural network, with both "pure" layers and "mix" layers. Options are also included for dropout, number of layers, and number of filters. We didn't use this model in our analysis, but its intended use it for convolving over latitudes, longitudes, and through time.
   - `CNN_RAPID.py`: A custom 2-dimensional convolutional neural network that adds an encoding for RAPID data which is concatenated after the convolutions are applied to surface variables. Otherwise the same as `CNN2D.py`
- **Sequence models:**
   - `RNN.py`: A basic recurrent neural network architecture, which is used in full Southern Ocean training. Options are included for dropout, number of hidden layers, and lengths of input/output streams.
   - `LSTM.py`: An implementation of a long short-term memory network, which is used in full Southern Ocean training. Options are included for dropout, number of hidden layers, and lengths of input/output streams.
   - `GRU.py`: An implementation of a gated recurrent unit, which is used in full Southern Ocean training. Options are included for dropout, number of hidden layers, and lengths of input/output streams.
- **Miscellaneous:**
   - `ESN.py`: An implementation of an Echo State Network, which is a fully-autoregressive deep learning model used in dynamical systems theory. We didn't use this model in our analysis, but its intended use was to help in predicting circulatory tipping points.

**PyTorch dataset definitions:**
- `SimDataset.py`: A minimal dataset wrapper for use with a `PyTorch` dataloader.
- `RAPIDDataset.py`: An extension of `SimDataset.py` that accounts for the auxiliary RAPID input.

**Utility functions:**
- `utils.py`: Many utility functions to facilitate modelling--data preprocessing, calculating metrics, and saving results.
- `plotting_utils.py`: Two helpful plots for visually assessing model performance in reconstructing the MOC strength.

**Model training loops:**
- `train.py`: A common training loop to be used for all neural network models. A mean squared error loss is used and AdamW is used for optimizing the model weights. Functionality is also provided for saving the model weights and training curve.

-----

### `notebooks/`

This directory contains all of our major data processing and all modelling experiments for ECCO.

**Data processing & exploration:**
- `streamfunction/`: A number of notebooks used to calculate depth- and density-space overturning streamfunctions (`psi_*.ipynb` and `sf_*.ipynb`), as well as explore the vertical profiles of the resulting streamfunctions (`plotting_streamfunctions.ipynb`)
- `moc/`: The notebooks used to calculate final MOC strength time series (`sl_moc.ipynb` and `so_moc.ipynb`) and plot them (`so_moc.ipynb` and `so_visualization.ipynb`).

**Replication of \[1\]:**
- `solodoch_replication/`: Contains all experiments (`train_models.ipynb`) and necessary utility functions (`replication_utils.py`) for replicating \[1\] using the neural network architecture defined in `models/SOLODOCH.py`. Experiments are performed with/without trend and seasonality.

**Linear regression experiments:**
- `linear_regression.ipynb`: All (regularised) linear regression experiments for the four latitudes of interest. Models can be fit as static in time vs. history of input variables, zonal averages vs. full zonal information, and with vs. without trend and seasonality.
- `latitude_transfer_linear_regression`: Experiments for (regularised) linear regression across the Southern Ocean. This includes both model transfer from 60S to all Southern Ocean latitudes as well as training an independent model on each Southern Ocean latitude. Much of this code is copied over from `linear_regression.ipynb`, but with minimal changes.

**CNN experiments:**
- `neural_networks.ipynb`: All experiments for 1-dimensional and 2-dimensional convolutional neural networks, with a focus on 60S. This notebook includes experiments with differing lengths of input history and model performance on each of the four latidues of interest.
- `Trend_and_season_neural_networks.ipynb`: 1-dimensional and 2-dimensional convolutional neural network experiments on the four latitudes of interest with different combinations of time series preprocessing (with/without trend and seasonality).

**RAPID experiments:**
- `RAPID_transfer_linear_regression.ipynb` and `RAPID_transfer_neural_network.ipynb`: Experiments to test the utility of integrating RAPID observational MOC strength data when predicting at southern latitudes (i.e., 30S) using (regularised) linear regression and convolutional neural networks, respectively.

**Full Southern Ocean experiments:**
- `southern_ocean_modelling/[MODEL].ipynb`: Model training on the _full_ Southern Ocean and evaluation across all Southern Ocean latitudes. The implements methods are: linear maps, multi-layer perceptrons, recurrent neural networks, long short-term memory networks, and gated recurrent units. All notebooks have the same overall structure.

-----

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
- `data_retrieval.ipynb`: Used to retrieve ACCESS data from the `data/` folder, preprocess input variables, and calculate the MOC strength time series. These `xarray` dataarrays are then saved to the `processed_data/` folder.
- `models.ipynb`: Used to construct and fit machine learning models for our final report. Models include (regularised) linear regression, multi-layer perceptrons, echo state networks, gaussian process regression, and XGBoost. Data required to run this notebook is produced by the `data_retrieval.ipynb` notebook.

-----

## Reproducing Report Figures and Tables

See the tables below for the notebooks to run to reproduce each figure and table in the final report. As a general case, these notebooks are contained within the `notebooks/` directory; however, all ACCESS-related results are in the `ACCESS/` directory (Tables 12-14).

### Figures

| Figure # |                    Notebook                     |
|:--------:|:-----------------------------------------------:|
|     1    |           `archive/basin_masks.ipynb`           |
|     2    |                        -                        |
|     3    | `streamfunction/plotting_streamfunctions.ipynb` |
|     4    | `streamfunction/plotting_streamfunctions.ipynb` |
|     5    |                        -                        |
|     6    | `streamfunction/plotting_streamfunctions.ipynb` |
|     7    | `streamfunction/plotting_streamfunctions.ipynb` |
|     8    |                        -                        |
|     9    |            `linear_regression.ipynb`            |
|    10    |    `solodoch_replication/train_models.ipynb`    |
|    11    |    `solodoch_replication/train_models.ipynb`    |
|    12    |             `neural_networks.ipynb`             |
|    13    |             `neural_networks.ipynb`             |
|    14    |    `RAPID_transfer_linear_regression.ipynb`     |
|    15    |   `latitude_transfer_linear_regression.ipynb`   |
|    16    |   `latitude_transfer_linear_regression.ipynb`   |
|    17    |      `southern_ocean_modelling/MLP.ipynb`       |
|    18    |             `neural_networks.ipynb`             |

### Tables

**Note:** as a general case, one cannot obtain the full results needed to reproduce each table with one run of the notebook. Instead, the user must change clearly marked hyperparameters, input variables, and preprocessing steps to sequentially obtain all results.

| Table # |                  Notebook                 |
|:-------:|:-----------------------------------------:|
|    1    |         `linear_regression.ipynb`         |
|    2    |         `linear_regression.ipynb`         |
|    3    | `solodoch_replication/train_models.ipynb` |
|    4    |  `Trend_and_season_neural_networks.ipynb` |
|    5    |          `neural_networks.ipynb`          |
|    6    |          `neural_networks.ipynb`          |
|    7    |          `neural_networks.ipynb`          |
|    8    |                     -                     |
|    9    |  `RAPID_transfer_linear_regression.ipynb` |
|    10   |   `RAPID_transfer_neural_network.ipynb`   |
|    11   |        `southern_ocean_modelling/*`       |
|    12   |           `ACCESS/models.ipynb`           |
|    13   |           `ACCESS/models.ipynb`           |
|    14   |           `ACCESS/models.ipynb`           |
|    15   |         `linear_regression.ipynb`         |
|    16   |         `linear_regression.ipynb`         |
|    17   |                     -                     |
|    18   |  `RAPID_transfer_linear_regression.ipynb` |
|    19   |   `RAPID_transfer_neural_network.ipynb`   |
|    20   |                     -                     |

-----

# References
[1] Solodoch, A., Stewart, A. L., McC. Hogg, A., & Manucharyan, G. E. (2023). Machine Learning‐Derived Inference of the Meridional Overturning Circulation From Satellite‐Observable Variables in an Ocean State Estimate. _Journal of Advances in Modeling Earth Systems_, 15(4), e2022MS003370.

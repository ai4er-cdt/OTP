## Directory containing scripts and notebooks required for processing data from the ACCESS-OM-01 dataset.

### Notebooks

- `data_retrieval.ipynb` is used to retrieve ACCESS data from the `data/` folder, preprocess input variables, and calculate the MOC. These Xarray dataarrays are then saved to the `processed_data/` folder.
- `models.ipynb` is used to construct and fit machine learning models in order to produce the results presented in Section **blah** of our final report. Models include (regularised) linear regression, multi-layer perceptrons, echo state networks, gaussaian process regression, and XGBoost. Data required to run this notebook is produced by the `data_retrieval.ipynb` notebook.


### Scripts

- `ESN.py` contains the constructor class for the Echo State Network architecture.
- `MLP.py` contains the constructor class for the Multilayer Perceptron architecture, as used to reproduce the results of Solodoch *et. al* (2023).
- `plotting_utils.py` contains functions required to generate timeseries comparison plots as shown in the final report, as well as scatter plots to visualise the performance of regression techniques.
- `utils.py` provides preprocessing functions that are customised to act on ACCESS data, including code to remove the trend and seasonality using an additive model.
- `SimDataset.py` provides a custom Pytorch DataSet class for our project, which is compatible with our training procedure.
- `train.py` defines the training function used to fit MLP models to ACCESS data, incorporating early-stopping, weight decay and model validation.


### Data folders

- `data/` contains relevant raw data taken from ACCESS-OM-01 model outputs. Variables include sea surface height, ocean bottom pressure, sea surface temperature, sea surface salinity, temperature fluxes, and horizontal ocean velocity fields.
- `processed_data/` contains processed input data for model development and training, as well as MOC strength timeseries calculated for 30S and 60S.


### Output folders

- `saved_models/` contains pytorch `state_dict` data used to save and reload neural network models.
- `loss_curves/` contains training loss curves for models trained by the `models.ipynb` notebook.
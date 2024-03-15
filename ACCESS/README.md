## Directory containing data and notebooks required for processing data from the ACCESS-OM-01 dataset.

- `data/` contains relevant data taken from ACCESS-OM-01 model outputs. Variables include sea surface height, ocean bottom pressure, sea surface temperature, sea surface salinity, temperature fluxes, and horizontal ocean velocity fields.
- `JRA55 data/` contains data retrieved from the JRA55 reanalysis product ([text](https://jra.kishou.go.jp/JRA-55/index_en.html)).
- `loss_curves/` contains training loss curves for models trained by the `models.ipynb` notebook.
- `processed_data/` contains preprocessed input data for model development and training, as well as MOC strength timeseries calculated for 30S and 60S.
- `saved_models/` contains pytorch `state_dict` data used to save and reload neural network models.
- `data_retrieval.ipynb` is used to retrieve ACCESS data from the `data/` folder, preprocess input variables, and calculate the MOC. These Xarray dataarrays are then saved to the `processed_data/` folder.
- `ESN.py` contains the constructor class for the Echo State Network architecture.
- `MLP.py` contains the constructor class for the Multilayer Perceptron architecture, as used to reproduce the results of Solodoch *et. al* (2023).
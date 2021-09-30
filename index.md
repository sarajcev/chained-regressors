## Machine Learning in Power System Analysis

Day-ahead forecasting of PV production using historical time-series data and weather information. 

Both single-step and multi-step forecasting of PV production with and without weather data is provided. Multi-step forecasting uses regressor chaining from `scikit-learn` with regressors (e.g. support vector machine) that do not support natively the multi-output regression.

### Dataset

Dataset comes from Kaggle and consists of two parts: (1) 5-second resolution PV production time-series and (2) 15-minute resolution time-series weather data.

The weather data is composed from the following weather variables:

- CD = low clouds (0 to 1)
- CM = medium clouds (0 to 1)
- CU = high clouds (0 to 1)
- PREC = precipitation (mm / 15 min)
- RH2m = relative humidity (%)
- SNOW = snow height (mm)
- ST = Surface Temperature (°C)
- SWD = Global Horizontal Irradiance (W/m2)
- SWDtop = Total Solar Irradiance at the top of the atmosphere (W/m2)
- TT2M = temperature 2 meters above the ground (°C)
- WS100m = Wind speed at 100m from the ground (m/s)
- WS10m = Wind speed at 10m from the ground (m/s)

### ML models

Different single-step and multi-step forecasting models are proposed and compared. Specifically, several different multi-step forecasting models are examined: 

- chained support vector machine regressor
- multi-output support vector machine regressor
- random forest (multi-output) regressor
- decision trees (multi-output) regressor

Furthermore, principal components analysis (PCA) can be used for features reduction. Standard scaler can be employed on the input data. 

Hyperparameters optimization is tackled using two different approaches: (a) RandomizedSearchCV and (b) HalvingRandomSearchCV, with cross-validation on the training set. Models use TimeSeriesSplit with cross-validation.

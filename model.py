#!/usr/bin/env python
# coding: utf-8

# ## Time-series forecasting of PV production

import timeit
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    # Seaborn style (figure aesthetics only)
    sns.set(context='paper', style='whitegrid', font_scale=1.2)
    sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})
except ImportError:
    print('Seaborn not installed. Going without it.')

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA

try:
    # Using experimental HalvingRandomSearchCV for hyperparameters optimization
    from sklearn.experimental import enable_halving_search_cv # noqa
    from sklearn.model_selection import HalvingRandomSearchCV
except ImportError:
    print('HalvingRandomSearchCV not found. Update scikit-learn to 0.24.')

from scipy import stats

import statsmodels.api as sm


# ### PV Data
# 
# 5-second resolution MiRIS PV from 13/05/2019 to 21/06/2019.
pv = pd.read_csv('miris_pv.csv', index_col=0, parse_dates=True)

# Resampling the dataset from 5-seconds to 15-minutes resolution (using mean)
pv = pv.resample('15min').mean()

# ### Weather Data

# 15-minute resolution weather data
# 
# The file is composed of forecast of several weather variables:
# 
#     CD = low clouds (0 to 1)
#     CM = medium clouds (0 to 1)
#     CU = high clouds (0 to 1)
#     PREC = precipitation (mm / 15 min)
#     RH2m = relative humidity (%)
#     SNOW = snow height (mm)
#     ST = Surface Temperature (°C)
#     SWD = Global Horizontal Irradiance (W/m2)
#     SWDtop = Total Solar Irradiance at the top of the atmosphere (W/m2)
#     TT2M = temperature 2 meters above the ground (°C)
#     WS100m = Wind speed at 100m from the ground (m/s)
#     WS10m = Wind speed at 10m from the ground (m/s)

we = pd.read_csv('weather_data.csv', index_col=0, parse_dates=True)


# ### Cleaning data

# Dropping SNOW and SWDtop from the dataset
we.drop('SNOW', axis=1, inplace=True)
we.drop('SWDtop', axis=1, inplace=True)

# Joining pv production and weather data into single dataframe
df = pd.concat([pv, we], axis=1)
df.dropna(inplace=True)
df.head()

# PV production and surface temp. data series plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4), 
                       gridspec_kw={'width_ratios': [3, 1]})
df.plot(y='PV', ax=ax[0])
df.plot(y='ST', lw=2, ax=ax[0], secondary_y=True)
ax[0].set_ylabel('PV')
ax[0].right_ax.set_ylabel('Temp (°C)')
sns.regplot(x='PV', y='ST', data=df, ax=ax[1], color='seagreen', 
            line_kws={'color':'darkgreen', 'linewidth':2},
            scatter_kws={'alpha':0.25})
ax[1].set_xlabel('PV')
ax[1].set_ylabel('')
fig.tight_layout()
plt.show()


def hampel_filter(input_series, window_size, 
                  scale_factor=1.4826, n_sigmas=3, 
                  overwrite=True, copy_series=True):
    """ Hampel filter for time-series data outlier detection and removal.

    Arguments
    ---------
    input_series: pd.Series
        Pandas series holding the original (unfiltered) time-series data.
    window_size: int
        Window size for the rolling window statistics of the Hampel filter.
    scale_factor: float
        Scale factor for the Hampel filter. Default value is provided with
        the assumption of the Gaussian distribution of samples.
    n_sigmas: int
        Number of standard deviations from the median, above/below which 
        a data point is marked as outlier. Default value is set at three 
        standard deviations.
    overwrite: bool
        True/False indicator for overwriting the outliers with the median 
        values, from rolling window statistics.
    copy_series: bool
        True/False indicator for making a local copy of the pandas series
        inside the function.

    Returns
    -------
    series: pd.Series
        New pandas series with outliers replaced (if overwrite=True) with 
        rolling median values; otherwise, original pd.series object.
    indices: list
        List of indices where outliers have been detected (and replaced 
        if overwrite=True) in the returned pd.series object 'series'. 
        This list can be empty if there were no outliers detected.

    Notes
    -----
    Hampel filter is used for detecting outliers in the time-series data
    (and their replacement with the rolling-window median values). It is 
    based on the rolling window statistics of the time-series values. It 
    flags as outliers any value that lies more than "n_sigmas" from the 
    median value, calculated using the rolling window approach.
    """
    if copy_series:
        series = input_series.copy()
    else:
        series = input_series
    
    # Median absolute deviation function 
    mad = lambda x: np.median(np.abs(x - np.median(x)))

    # Rolling statistics
    rolling_median = input_series.rolling(window=2*window_size, center=True).median()
    difference = np.abs(input_series - rolling_median)
    rolling_mad = scale_factor * input_series.rolling(window=2*window_size, center=True).apply(mad)
    indices = list(np.argwhere(difference.values > (n_sigmas * rolling_mad.values)).flatten())

    # Overwriting outliers with rolling median values
    if len(indices) == 0:
        print('There were no outliers found within {:d} standard deviations.'.format(n_sigmas))
    else:
        print('Found {:d} outliers within {:d} standard deviations.'.format(len(indices), n_sigmas))
        if overwrite:
            # Overwrite outliers with rolling median values
            print('Overwriting outliers with rolling median values!')
            series[indices] = rolling_median[indices] 

    return series, indices


# Apply Hampel filter
filtered, outliers = hampel_filter(df['ST'], window_size=8)

# Plot outliers
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['ST'].index, df['ST'].values, lw=1, c='seagreen', label='ST', zorder=1)
ax.scatter(df['ST'].index[outliers], df['ST'].values[outliers], 
           marker='o', c='darkred', s=25, label='outliers', zorder=2)
ax.scatter(filtered.index[outliers], filtered.values[outliers], 
           marker='s', c='navy', s=20, label='corrected', zorder=2)
ax.set_ylabel('Temp (°C)')
ax.grid(which='major', axis='y')
ax.legend(loc='best')
fig.tight_layout()
plt.show()


def plot_correlations(dataframe, column_names, lags=24, 
                      copy_data=True, resample=True):
    """ Autocorrelation (ACF), Partial autocorrelation (PACF) and 
    Cross-correlation (CCF) plots of two different time-series.

    Arguments
    ---------
    dataframe: pd.DataFrame
        Pandas dataframe holding the original time-series data 
        (in the long table format).
    column_names: list
        List of two column names from the dataframe which provide time-series
        data for creating the ACF, PACF and CCF plots.
    lags: int
        Time lags used for the autocorrelation, partial autocorrelation and 
        cross-correlation plots.
    copy_data: bool
        True/False indicator for making a local copy of the dataframe
        inside the function.
    resample: bool
        True/False indicator for resampling data to hourly frequency.
   
    Notes
    -----
    Function displays a matplotlib plot of the time-series data, histograms, as
    well as the autocorrelation, partial autocorrelation and cross-correlation 
    of the selected time-series variables. These figures can aid in determining
    the most appropriate time shifts (lags) for the features engineering.
    """
    if copy_data:
        df = dataframe.copy()
    else:
        df = dataframe
    if resample:
        df = df.resample('1H').mean()
    
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(10,8))
    gs = ax[4,0].get_gridspec()
    ax[4,0].remove(); ax[4,1].remove()
    ax4 = fig.add_subplot(gs[4,:])
    df[column_names[0]].plot(ax=ax[0,0], title='Variable: '+column_names[0])
    df[column_names[1]].plot(ax=ax[0,1], title='Variable: '+column_names[1])
    df[column_names[0]].plot.hist(bins=12, ax=ax[1,0])
    df[column_names[1]].plot.hist(bins=12, ax=ax[1,1])
    sm.graphics.tsa.plot_acf(df[column_names[0]], ax=ax[2,0], lags=24, title='Autocorrelation')
    sm.graphics.tsa.plot_pacf(df[column_names[0]], ax=ax[3,0], lags=24, title='Partial autocorrelation')
    sm.graphics.tsa.plot_acf(df[column_names[1]], ax=ax[2,1], lags=24, title='Autocorrelation')
    sm.graphics.tsa.plot_pacf(df[column_names[1]], ax=ax[3,1], lags=24, title='Partial autocorrelation')
    for axis in ax.flatten()[4:]:
        axis.set_xlabel('Time lag (hours)')
    ax[2,0].set_ylabel('ACF')
    ax[3,0].set_ylabel('PACF')
    ccf = sm.tsa.stattools.ccf(df[column_names[0]], df[column_names[1]])
    ax4.plot(ccf[:lags])
    ax4.set_title('Cross-correlation between {} and {}'.format(column_names[0], column_names[1]))
    ax4.set_xlabel('Time lag (hours)')
    ax4.set_ylabel('CCF')
    fig.tight_layout()
    plt.show()
    return

# Show plots
plot_correlations(df, column_names=['PV', 'ST'])


# Pearson correlation between PV production and surface temperature
print('Pearson correlation between PV production and surface temperature:')
print(df[['PV', 'ST']].corr())


# ### Features engineering from the time-series data

def engineer_features(dataframe, window=24, steps_ahead=1, 
                      copy_data=True, resample=True, drop_nan_rows=True,
                      weather_data=True):
    """ Engineer features from the time-series data.

    Features engineering from the time-series data by using time-shift,
    first-differencing, rolling window statistics, cyclical transforms,
    and encoding. NaN values are dropped from the final dataset.

    NOTE: Function is tailored for the hourly sampled data time-series!

    Arguments
    ---------
    dataframe: pd.DataFrame
        Pandas dataframe holding the original time-series data 
        (in the long table format).
    window: int
        Window size for the past observations (for time-shifting).
    steps_ahead: int
        Number of time steps ahead for multi-step forecasting 
        (steps_ahead=1 means single-step ahead forecasting).
    copy_data: bool
        True/False indicator for making a local copy of the dataframe
        inside the function.
    resample: bool
        True/False indicator for resampling data to hourly frequency.
    drop_nan_rows: bool
        True/False indicator to drop rows with NaN values.
    weather_data: bool
        True/False indicator for using weather information during
        features engineering.

    Returns
    -------
    df: pd.DataFrame
        Pandas dataframe with newly engineered features (long format).
    """
    if copy_data:
        df = dataframe.copy()
    else:
        df = dataframe
    if resample:
        df = df.resample('1H').mean()
    
    # Engineer features from time-series data
    if weather_data:
        columns = df.columns
        for col in columns:
            for i in range(1, window+1):
                # Shift data by lag of 1 to window=24 hours
                df[col+'_{:d}h'.format(i)] = df[col].shift(periods=i)  # time-lag
        for col in columns:
            df[col+'_diff'] = df[col].diff()  # first-difference
    else:
        # Additional features only for PV only (weather data is completely unused)
        for i in range(1, window+1):
            # Shift data by lag of 1 to window=24 hours
            df['PV'+'_{:d}h'.format(i)] = df['PV'].shift(periods=i)  # time-lag
        df['PV_diff'] = df['PV'].diff()    
    df['PV_diff24'] = df['PV'].diff(24)

    # Rolling windows (24-hours) on time-shifted PV production
    df['roll_mean'] = df['PV_1h'].rolling(window=24, win_type='hamming').mean()
    df['roll_max'] = df['PV_1h'].rolling(window=24).max()
    
    # Hour-of-day indicators with cyclical transform
    dayhour_ind = df.index.hour
    df['hr_sin'] = np.sin(dayhour_ind*(2.*np.pi/24))
    df['hr_cos'] = np.cos(dayhour_ind*(2.*np.pi/24))
    
    # Month indicators with cyclical transform
    month_ind = df.index.month
    df['mnth_sin'] = np.sin((month_ind-1)*(2.*np.pi/12))
    df['mnth_cos'] = np.cos((month_ind-1)*(2.*np.pi/12))

    # Encoding sunshine hours
    sun_ind = df['PV'] > 0.
    df['sun'] = sun_ind.astype(int)

    # Forecast horizont
    if steps_ahead == 1:
        # Single-step ahead forecasting
        df['PV+0h'] = df['PV'].values
    else:
        # Multi-step ahead forecasting
        for i in range(steps_ahead):
            df['PV'+'+{:d}h'.format(i)] = df['PV'].shift(-i)
    del df['PV']

    # Drop rows with NaN values
    if drop_nan_rows:
        df.dropna(inplace=True)

    return df


# Single-step model
df2 = engineer_features(df)


# ### Train, validation, and test datasets (time-series data)

def prepare_data(dataframe, weather_forecast=False, copy_data=True):
    """ Prepare dataframe for spliting into train and test sets.

    Arguments
    ---------
    dataframe: pd.DataFrame
        Pandas dataframe holding the original time-series data 
        (in the long table format).
    weather_forecast: bool
        True/False variable indicating if hour-ahead weather 
        forecast is available or not when making predictions.
    copy_data: bool
        True/False indicator for making a local copy of the dataframe
        inside the function.

    Returns
    -------
    X, y: pd.DataFrame
        Matrix X of features and vector (matrix) y of targets.
    """
    if copy_data:
        df = dataframe.copy()
    else:
        df = dataframe
    if weather_forecast is False:
        # Hour-ahead weather forecast is NOT being utilized
        df.drop(columns=['CD', 'CM', 'CU', 'PREC', 'RH2m', 'ST', 
                        'SWD', 'TT2M', 'WS100m', 'WS10m'],
                inplace=True)

    columns = df.columns.values
    outputs = [col_name for col_name in columns if 'PV+' in col_name]
    inputs = [col_name for col_name in columns if col_name not in outputs]
    # inputs (features)
    X = df[inputs]
    # outputs
    y = df[outputs]
    return X, y


def exponential_sample_weights(num, shape=1.):
    """ Generating exponential sample weights.

    Parameters
    ----------
    num: int
        Number of samples.
    shape: float
        Number indicating the steepness of the exponential function.
        Larger number means larger steepness. Usually, a number in 
        the range 1 to 5 would be enough to put the weight on the
        most-recent samples.

    Returns
    -------
    weight: 1D array (num)
        Sample weights from the exponential function, starting from
        the largest weight at the position weight[0] and decreasing.

    Notes
    -----
    It is assumed that the samples array for which the weights are  
    being generated here constitutes an ordered time-series, starting 
    with the most-recent sample at the position zero!
    """
    indices = np.linspace(0., shape, num=num)
    sample_weights = np.exp(-indices)
    return sample_weights


# ### Walk-forward multi-step prediction with a single-step model

WALK = 12  # walk-forward for WALK hours
STEP = 24  # multi-step predict for STEP hours ahead
# With STEP=24 and WALK=12, we are making a 24-hour ahead predictions 
# after each hour, and move forward in time for 12 hours in total. 
# In other words, we walk forward for 12 hours, and each time we move 
# forward (by one hour) we make a brand new 24-hour ahead predictions. 
# Predicted values are being utilized as past observations for making
# new predictions as we walk forward in time. Hence, as we move away in 
# time from the present moment we are relying more and more on predicted 
# values to make new predictions!


def walk_forward(X_values, y_predicted, window=24, weather_forecast=False):
    """ Walk forward

    Preparing input matrix X for walk-forward single-step predictions.
    There are eleven different original variables (PV plus 10 weather 
    vars.), which have been time-shifted using the "window" method.

    NOTE: Function uses certain hard-coded elements, specially adjusted
          for the particular problem/ dataset at hand. 

    Arguments
    ---------
    X_values: np.array
        Input values for making predictions.
    y_predicted: float
        Predicted value.
    window: int
        Window size of the past observations. It should be equal to 
        the window size that was used for features engineering.
    weather_forecast: bool
        True/False variable indicating if hour-ahead weather 
        forecast is available or not when making predictions.

    Returns
    -------
    X_values: np.array
        Input values time-shifted by a single time step, using the 
        walk forward approach.

    Raises
    ------
    NotImplementedError
        Walk forward is not implemented with hour-ahead weather forecast.
    """
    #TODO: Implement a walk forward with the hour-ahead weather forecast.
    if weather_forecast:
        raise NotImplementedError('Walk forward is not implemented with hour-ahead weather forecast.')
    
    # There are eleven different original
    # variables (PV plus 10 weather vars)
    X_parts = []
    j = 0; k = 0
    for i in range(11):
        k = j + window
        X_part = X_values[j:k]
        X_part = pd.Series(X_part)
        if i == 0:
            # time-shifted PV production
            X_part = X_part.shift(periods=1, fill_value=y_predicted)
        else:
            # time-shifted weather features
            X_part = X_part.shift(periods=1, fill_value=np.NaN)
            X_part.fillna(method='bfill', inplace=True)  # back-fill
        X_parts.append(X_part.values)
        j += window
    X_parts = np.asarray(X_parts).reshape(1,-1).flatten()
    X_rest = X_values[-19:]   # other features (hard-coded)
    # Update feature PV_diff with y_predicted
    X_rest[0] = X_parts[0] - X_parts[1]
    # Concatenate
    X_values = np.r_[X_parts, X_rest]
    return X_values


def plot_predictions(walk, y_test, y_pred):
    plt.figure(figsize=(6,4))
    plt.title('walk forward +{:2d} hours'.format(walk+1))
    plt.plot(y_test.values[walk:walk+STEP], lw=2.5, label='true values')
    plt.plot(y_pred, ls='--', lw=1.5, marker='+', ms=10, label='predictions')
    mae = mean_absolute_error(y_test.values[walk:walk+STEP], y_pred)
    plt.text(STEP-2, 0.35, 'MAE: {:.3f}'.format(mae), 
             horizontalalignment='right', 
             fontweight='bold')
    plt.legend(loc='upper right')
    plt.ylim(top=0.5)
    plt.grid(axis='y')
    plt.xlabel('Hour')
    plt.ylabel('PV power')
    plt.show()


# Single step model prediction
single_step_model = False

if single_step_model:
    # Hour-ahead weather forecast is NOT being utilized
    weather_forecast = False
    # Prepare dataframe for a split into train, test sets
    X, y = prepare_data(df2)
    # Train and test dataset split (w/o shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    # Print train and test set shapes
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    model = 'RandomForest' # 'AdaBoost'
    
    if model == 'RandomForest':
        # Pipeline: SelectKBest and RandomForest
        # SelectKBest is used for features selection/reduction
        selectbest = SelectKBest(score_func=mutual_info_regression, k='all')
        forest = RandomForestRegressor(criterion='mse', bootstrap=True)
        # Creating a pipeline
        pipe = Pipeline(steps=[('preprocess', 'passthrough'), 
                               ('kbest', selectbest), 
                               ('estimator', forest)])
        # Parameters of pipeline for the randomized search with cross-validation
        param_dists = {'preprocess': [None, StandardScaler()], 
                       'kbest__k': stats.randint(low=32, high=128), 
                       'estimator__n_estimators': stats.randint(low=200, high=1000),
                       'estimator__max_depth': [1, 3, 5, None], 
                       'estimator__max_samples': stats.uniform(loc=0.2, scale=0.8),
                      }
    elif model == 'AdaBoost':
        # AdaBoost with Decision tree regressor (supports multi-output natively)
        ada = AdaBoostRegressor(base_estimator=DecisionTreeRegressor())
        # Creating a pipeline
        pipe = Pipeline(steps=[('estimator', ada)])
        # Parameters of pipeline for the randomized search with cross-validation
        param_dists = {# Hyper-parameters of the base estimator (DecisionTree)
                       'estimator__base_estimator__max_depth': [3, 4],  # three-levels deep
                       'estimator__base_estimator__min_samples_leaf': [1, 2],
                       # Hyper-parameters of the ensemble estimator (AdaBoost)
                       'estimator__n_estimators': stats.randint(low=50, high=500),
                       'estimator__learning_rate': stats.uniform(1e-2, 1e1),
                       }
    elif model == 'SVR':
        # Pipeline: SelectKBest and SVR
        # SelectKBest is used for features selection/reduction
        selectbest = SelectKBest(score_func=mutual_info_regression, k='all')
        # Support Vector Regression 
        svr = SVR(kernel='rbf', gamma='scale')
        # Creating a pipeline
        pipe = Pipeline(steps=[('preprocess', 'passthrough'), 
                               ('kbest', selectbest), 
                               ('estimator', svr)])
        # Parameters of pipeline for the randomized search with cross-validation
        param_dists = {'preprocess': [None, StandardScaler()], 
                       'kbest__k': stats.randint(low=32, high=128), 
                       'estimator__C': stats.loguniform(1e0, 1e3),
                       'estimator__epsilon': stats.loguniform(1e-5, 1e-2),
                      }
    else:
        raise NotImplementedError('Model name "{}" is not recognized or implemented!'.format(model))

    NITER = 100  # number of random search iterations
    NJOBS = 7    # Determine the number of parallel jobs

    sample_weighting = True  # use sample weighting

    time_start = timeit.default_timer()
    search = RandomizedSearchCV(estimator=pipe, param_distributions=param_dists, 
                                cv=TimeSeriesSplit(n_splits=3),
                                scoring='neg_mean_squared_error',
                                n_iter=NITER, refit=True, n_jobs=NJOBS)
    if sample_weighting:
        # Exponentially weighting samples (emphasis on the most recent ones)
        sample_weights = exponential_sample_weights(X_train.shape[0], 2.)
        search.fit(X_train, y_train, estimator__sample_weight=sample_weights)
    else:
        search.fit(X_train, y_train)
    time_end = timeit.default_timer()
    time_elapsed = time_end - time_start
    print('Execution time (hour:min:sec): {}'.format(str(dt.timedelta(seconds=time_elapsed))))
    print('Best parameter (CV score = {:.3f}):'.format(search.best_score_))
    print(search.best_params_)
    
    if model == 'RandomForest':
        # Feature importance analysis 
        best_params = {'n_estimators': search.best_params_['estimator__n_estimators'],
                       'max_depth': search.best_params_['estimator__max_depth'],
                       'max_samples': search.best_params_['estimator__max_samples'],
                      }
        forest = RandomForestRegressor(criterion='mse', **best_params)
        forest.fit(X_train, y_train)

        TOP = 15
        feature_importance = forest.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)[-TOP:]
        pos = np.arange(sorted_idx.shape[0]) + .25

        # Plot relative feature importance
        fig, ax = plt.subplots(figsize=(7,5))
        ax.barh(pos, feature_importance[sorted_idx][-TOP:],
                align='center', color='magenta', alpha=0.6)
        plt.yticks(pos, X_train.columns[sorted_idx][-TOP:])
        ax.set_xlabel('Feature Relative Importance')
        ax.grid(axis='x')
        plt.tight_layout()
        plt.show()
    
    # Make single-step predictions for 24 hours ahead
    y_preds = search.predict(X_test.values[:24,:])
    
    mse = mean_squared_error(y_test.values[:24], y_preds)
    print('MSE:', mse.round(5))
    mae = mean_absolute_error(y_test.values[:24], y_preds)
    print('MAE:', mae.round(5))
    
    plt.figure(figsize=(6,4))
    plt.plot(y_test.index[:24], y_test.values[0:24], lw=2, label='true values')
    plt.plot(y_test.index[:24], y_preds, ls='--', lw=1.5,
             marker='+', ms=10, label='predictions')
    plt.text(y_test.index[20], 0.35, 'MAE: {:.3f}'.format(mae),
             horizontalalignment='center', fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.xlabel('Day/Hour')
    plt.ylabel('PV power')
    plt.show()

    # Do walk-forward predictions (ONLY if weather_forecast == False)
    for k in range(WALK):
        X_test_values = X_test.values[k,:]
        y_pred_values = []
        for i in range(STEP):
            # Predict next time-step value
            y_predict = search.predict(X_test_values.reshape(1,-1))[0]
            y_pred_values.append(y_predict)
            # Walk-forward for a single time step
            X_test_values = walk_forward(X_test_values, y_predict)
        # Plot walk-forward predictions against true values
        plot_predictions(k, y_test, y_pred_values)


# ### Multi-step model pipeline without features selection

# Multi-step model (24-hours ahead)
df2 = engineer_features(df, steps_ahead=STEP, weather_data=False)
# Prepare dataframe for a split into train, test sets
X, y = prepare_data(df2)

# Train and test dataset split (w/o shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

# Print train and test set shapes
print('Train and test set data shapes:')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

multi_model = 'ChainSVR'

if multi_model == 'RandomForest':
    # Random Forest Regression (supports multi-output natively)
    forest = RandomForestRegressor(criterion='mse', bootstrap=True)
    # Creating a pipeline
    pipe = Pipeline(steps=[('preprocess', 'passthrough'), 
                           ('forest', forest)])
    # Parameters of pipeline for the randomized search with cross-validation
    param_dists = {'preprocess': [None, StandardScaler()], 
                   'forest__n_estimators': stats.randint(low=700, high=1000),
                   'forest__max_depth': [1, 3, 5, None], 
                   'forest__max_samples': stats.uniform(loc=0.2, scale=0.8),
                   'forest__min_samples_split': stats.randint(low=2, high=11),
                   'forest__min_samples_leaf': stats.randint(low=1, high=11),
                   'forest__max_features': [50, 100, 150, 200, 250, None],
                  }
elif multi_model == 'DecisionTree':
    # Decision tree regressor 
    tree = DecisionTreeRegressor()
    # Creating a pipeline
    pipe = Pipeline(steps=[('tree', tree)])
    param_dists = {'tree__criterion': ['mse', 'mae'],
                   'tree__max_depth': [2, 4, 6, 8, None],
                   'tree__min_samples_leaf': stats.randint(low=1, high=9),
                   }
elif multi_model == 'ChainSVR':
    # Support Vector Regression (does NOT support multi-output natively)
    svr = RegressorChain(SVR(kernel='rbf', cache_size=500))
    # Creating a pipeline
    pipe = Pipeline(steps=[('preprocess', 'passthrough'), 
                           ('svr', svr)])
    # Parameters of pipeline for the randomized search with cross-validation
    param_dists = {'preprocess': [None, StandardScaler()], 
                   'svr__base_estimator__C': stats.loguniform(1e0, 1e3),
                   'svr__base_estimator__epsilon': stats.loguniform(1e-5, 1e-2),
                   'svr__base_estimator__gamma': ['scale', 'auto'],
                  }                 
elif multi_model == 'MultiSVR':
    # Support Vector Regression (does NOT support multi-output natively)
    svr = MultiOutputRegressor(SVR(kernel='rbf', gamma='scale'))
    # Creating a pipeline
    pipe = Pipeline(steps=[('preprocess', 'passthrough'), 
                           ('svr', svr)])
    # Parameters of pipeline for the randomized search with cross-validation
    param_dists = {'preprocess': [None, StandardScaler()], 
                   'svr__estimator__C': stats.loguniform(1e0, 1e3),
                   'svr__estimator__epsilon': stats.loguniform(1e-5, 1e-2),
                  }
elif multi_model == 'PCA+SVR':
    # Principal Component Analysis (PCA) is used for decomposing 
    # (i.e. projecting) features into the lower-dimensional space
    # while retaining maximum amount of the variance.
    pca = PCA(whiten=True, svd_solver='full')
    # Support Vector Regression (does NOT support multi-output natively)
    svr = RegressorChain(SVR(kernel='rbf', cache_size=500))
    # Creating a pipeline
    pipe = Pipeline(steps=[('pca', pca), 
                           ('svr', svr)])
    # Parameters of pipeline for the randomized search with cross-validation
    param_dists = {'pca__n_components': stats.uniform(),
                   'svr__base_estimator__C': stats.loguniform(1e0, 1e3),
                   'svr__base_estimator__epsilon': stats.loguniform(1e-5, 1e-2),
                   'svr__base_estimator__gamma': ['scale', 'auto'],
                  }
else:
    raise NotImplementedError('Model name "{}" is not recognized or implemented!'.format(multi_model))

NITER = 100  # number of random search iterations
NJOBS = -1   # Determine the number of parallel jobs
print('Running ...')

# Choose a search method for hyperparameters optimization
search_type = 'HalvingRandomSearchCV'

if search_type == 'RandomizedSearchCV':
    time_start = timeit.default_timer()
    search_multi = RandomizedSearchCV(estimator=pipe, param_distributions=param_dists,
                                      cv=TimeSeriesSplit(n_splits=3),
                                      scoring='neg_mean_squared_error',
                                      n_iter=NITER, refit=True, n_jobs=NJOBS)
    search_multi.fit(X_train, y_train)
    time_end = timeit.default_timer()
    time_elapsed = time_end - time_start
    print('Execution time (hour:min:sec): {}'.format(str(dt.timedelta(seconds=time_elapsed))))
    print('Best parameter (CV score = {:.3f}):'.format(search_multi.best_score_))
    print(search_multi.best_params_)

elif search_type == 'HalvingRandomSearchCV':
    time_start = timeit.default_timer()
    search_multi = HalvingRandomSearchCV(estimator=pipe, param_distributions=param_dists,
                                         cv=TimeSeriesSplit(n_splits=3),
                                         scoring='neg_mean_squared_error',
                                         refit=True, n_jobs=NJOBS)
    search_multi.fit(X_train, y_train)
    time_end = timeit.default_timer()
    time_elapsed = time_end - time_start
    print('Execution time (hour:min:sec): {}'.format(str(dt.timedelta(seconds=time_elapsed))))
    print('Best parameter (CV score = {:.3f}):'.format(search_multi.best_score_))
    print(search_multi.best_params_)

else:
    raise NotImplementedError('Search method "{}" is not recognized or implemented!'.format(search_type))

def plot_multi_step_predictions(walk, y_test, y_pred):
    plt.figure(figsize=(6,4))
    plt.title('walk forward +{:2d} hours'.format(walk+1))
    plt.plot(y_test, lw=2.5, label='true values')
    plt.plot(y_pred, ls='--', lw=1.5, marker='+', ms=10, label='predictions')
    mae = mean_absolute_error(y_test, y_pred)
    plt.text(STEP-2, 0.35, 'MAE: {:.3f}'.format(mae), 
             horizontalalignment='right', 
             fontweight='bold')
    plt.legend(loc='upper right')
    plt.ylim(top=0.5)
    plt.grid(axis='y')
    plt.xlabel('Hour')
    plt.ylabel('PV power')
    plt.show()


# Do multi-step ahead predictions
for k in range(WALK):
    X_test_values = X_test.values[k+20,:]  # +20 hard-coded shift to align views with those
    y_test_values = y_test.values[k+20,:]  # of walk-forward predictions for easy comparison
    y_predict = search_multi.predict(X_test_values.reshape(1,-1)).flatten()
    # Manually correct (small) negative predicted values
    y_predict = np.where(y_predict < 0., 0., y_predict)
    # Plot multi-step predictions against true values
    plot_multi_step_predictions(k, y_test_values, y_predict)


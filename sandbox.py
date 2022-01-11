import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def hampel_filter(input_series, window_size, 
                  scale_factor=1.4826, n_sigmas=3, 
                  overwrite=True, copy_series=True):
    """ Hampel filter for time-series data outlier detection and removal

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
    flags as outliers any value that lies more than 'n_sigmas' from the 
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


def engineer_features(dataframe, window=24, steps_ahead=1, 
                      copy_data=True, resample=True, drop_nan_rows=True):
    """ Engineer features from time-series data

    Features engineering from the time-series data by using time-shift,
    first-differencing, rolling window statistics, cyclical transforms,
    and encoding. NaN values are dropped from the final dataset.

    NOTE: Function is tailored for the hourly sampled data time-series.

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
    columns = df.columns
    for col in columns:
        for i in range(1, window+1):
            # Shift data by lag of 1 to window=24 hours
            df[col+'_{:d}h'.format(i)] = df[col].shift(periods=i)  # time-lag
    for col in columns:
        df[col+'_diff'] = df[col].diff()  # first-difference
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


def prepare_data(dataframe, weather_forecast=False, copy_data=True):
    """ Prepare dataframe for spliting into train and test sets

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
    """ Generating exponential sample weights

    Parameters
    ----------
    num: int
        Number of samples.
    shape: float
        Number indicating the steepness of the exponential function.
        Larger number means larger steepness. Usually 1 to 5 would
        be enough to put the weight on the most-recent samples.

    Returns
    -------
    weight: 1D array (num)
        Sample weights from the exponential function, starting from
        the largest weight at the position weight[0] and decreasing.

    Notes
    -----
    It is assumed that the samples array for which the weights are  
    being generated here constitute an ordered time-series, starting 
    with the most-recent sample at the position zero.
    """
    indices = np.linspace(0., shape, num=num)
    sample_weights = np.exp(-indices)
    return sample_weights

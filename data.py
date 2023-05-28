import glob

import pandas as pd
import statsmodels.api as sm


def process_meters_data(resolution):
    # Get a list of all CSV files in the directory
    csv_files = glob.glob('CSV/meters/*.csv')

    # Initialize an empty list to store the DataFrames
    meters_df_list = []

    for file in csv_files:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file, names=['Date & Time', 'Total Usage [kW]'])

        # Convert 'Date & Time' to datetime
        df['Date & Time'] = pd.to_datetime(df['Date & Time'])

        # Convert 'total' column to float and replace NaNs with 0
        df['Total Usage [kW]'] = pd.to_numeric(df['Total Usage [kW]'], errors='coerce').fillna(0)

        # Rename 'Date & Time' column
        df.rename(columns={'Date & Time': 'Date'}, inplace=True)

        # Append the current DataFrame to the list
        meters_df_list.append(df)

    # Combine all meters dataframes into one
    meters_df = pd.concat(meters_df_list, ignore_index=True)

    # Sort the combined meters data by date
    meters_df.sort_values('Date', inplace=True)

    # Set 'Date' as the index for combined meters data
    meters_df.set_index('Date', inplace=True)

    # Resample the combined meters data
    meters_data_resampled = meters_df.resample(resolution).sum()

    # Interpolate NaNs
    meters_data_resampled.interpolate(method='quadratic', inplace=True)

    # Reset the index
    meters_data_resampled.reset_index(inplace=True)

    return meters_data_resampled


def process_weather_data(resolution):
    # Get a list of all CSV files in the directory
    csv_files = glob.glob('CSV/weather/*.csv')

    # Initialize an empty list to store the DataFrames
    weather_df_list = []

    for file in csv_files:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)

        # Drop unnecessary columns
        df = df.drop(columns=['icon', 'summary'])

        # Convert the 'time' column to datetime format and rename it to 'Date'
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Rename columns in one call
        df.rename(columns={
            'time': 'Date',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'apparentTemperature': 'Apparent Temperature',
            'visibility': 'Visibility',
            'pressure': 'Pressure',
            'windSpeed': 'Wind Speed',
            'cloudCover': 'Cloud Cover',
            'windBearing': 'Wind Bearing',
            'precipIntensity': 'Precipitation Intensity',
            'dewPoint': 'Dew Point',
            'precipProbability': 'Precipitation Probability'
        }, inplace=True)

        # Convert all non-date columns to float and replace NaN values with 0
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].astype(float).fillna(0)

        # Convert 'Temperature' from Fahrenheit to Celsius
        df['Temperature'] = (df['Temperature'] - 32) / 1.8

        # Append the current DataFrame to the list
        weather_df_list.append(df)

    # Combine all weather dataframes into one
    weather_df = pd.concat(weather_df_list)

    # Sort the combined meters data by date
    weather_df.sort_values('Date', inplace=True)

    # Set 'Date' as the index
    weather_df.set_index('Date', inplace=True)

    # Group by date and aggregate with average
    weather_df = weather_df.resample(resolution).mean()

    # Interpolate NaNs
    weather_df.interpolate(method='quadratic', inplace=True)

    return weather_df


def backward_elimination(df, dependent_col, protected_cols, alpha=0.05):
    # Initial set of predictors
    predictors = df.columns.tolist()
    predictors.remove(dependent_col)
    for col in protected_cols:
        predictors.remove(col)

    while len(predictors) > 0:
        predictors_with_constant = sm.add_constant(df[predictors])
        # Skip the intercept's p-value
        p_values = sm.OLS(df[dependent_col], predictors_with_constant).fit().pvalues[1:]
        # Get the max p-value
        max_p_value = p_values.max()

        if max_p_value > alpha:
            removed_predictor = p_values.idxmax()
            predictors.remove(removed_predictor)
        else:
            break

    return df[protected_cols].join(df[dependent_col]).join(df[predictors])


def get_significant_data(resolution):
    # Process the weather and meters data
    weather_data = process_weather_data(resolution)
    meters_data = process_meters_data(resolution)

    # Merge the two dataframes on 'Date'
    df = pd.merge(weather_data, meters_data, on='Date', how='outer')

    # Replace any NaN values with 0
    df.fillna(0, inplace=True)

    # Perform backward elimination
    minimum_df = backward_elimination(df, 'Total Usage [kW]', ['Date'], 0.05)

    return minimum_df

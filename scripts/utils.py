import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix


def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    return outliers

def plot_time_series(df):

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    fig, ax = plt.subplots(figsize=(10, 6)) 

    ax.plot(df['Timestamp'], df['GHI'], label='GHI', color='blue', linestyle='-') 
    ax.plot(df['Timestamp'], df['DNI'], label='DNI', color='green', linestyle='--')  
    ax.plot(df['Timestamp'], df['DHI'], label='DHI', color='orange', linestyle='-.')  
    ax.plot(df['Timestamp'], df['Tamb'], label='Tamb', color='red', linestyle=':')  

    ax.set_xlabel('Timestamp') 
    ax.set_ylabel('Values') 
    ax.set_title('Change of Variables over Time')
    ax.legend(loc='upper right')
    plt.setp(ax.get_xticklabels(), rotation=45) 
    fig.tight_layout() 
    plt.show() 




def plot_correlation_analysis(data_frame):
    # Limit the data to the first 1000 rows
    limited_data_frame = data_frame.head(1000)
    
    # Define columns related to solar radiation and temperature
    solar_temp_columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    wind_columns = ['WS', 'WSgust', 'WD']

    # Compute the correlation matrix for solar radiation and temperature columns
    correlation_matrix = limited_data_frame[solar_temp_columns].corr()

    # Plot the heatmap for the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap: Solar Radiation and Temperature')
    plt.show()

    # Generate a pair plot to visualize relationships between solar radiation and temperature measures
    sns.pairplot(limited_data_frame[solar_temp_columns])
    plt.suptitle('Pair Plot: Solar Radiation and Temperature Measures', y=1.02)
    plt.show()

    # Create a scatter matrix for wind measurements and solar irradiance
    scatter_matrix(limited_data_frame[wind_columns + ['GHI', 'DNI', 'DHI']], figsize=(12, 12), diagonal='kde', alpha=0.6)
    plt.suptitle('Scatter Matrix: Wind Conditions and Solar Irradiance', y=1.02)
    plt.show()

def plot_wind_analysis(data, ws_col='WS', wd_col='WD', title='Wind Speed and Direction Analysis'):
    # Convert wind direction from degrees to radians
    data['Wind_Direction_Radians'] = np.deg2rad(data[wd_col])

    # Create a polar plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # Scatter plot of wind speed and direction
    sc = ax.scatter(data['Wind_Direction_Radians'], data[ws_col], 
                    c=data[ws_col], cmap='viridis', alpha=0.75)

    # Add a colorbar for wind speed
    plt.colorbar(sc, label='Wind Speed (m/s)')

    # Set plot title and labels
    ax.set_title(title, va='bottom')
    ax.set_theta_zero_location('N')  # Set 0 degrees at North
    ax.set_theta_direction(-1)       # Set degrees to increase clockwise

    # Show the plot
    plt.show()

    # Analysis of variability in wind direction
    wind_direction_std_dev = data[wd_col].std()
    print(f"Wind Direction Variability (Standard Deviation): {wind_direction_std_dev:.2f} degrees")

def plot_temperature_analysis(data_frame):
    # Scatter plots: RH vs Temperature, RH vs Solar Radiation
    plt.figure(figsize=(16, 10))

    # RH vs TModA
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=data_frame['RH'], y=data_frame['TModA'])
    plt.title('Relative Humidity vs TModA')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Temperature (TModA)')

    # RH vs TModB
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=data_frame['RH'], y=data_frame['TModB'])
    plt.title('Relative Humidity vs TModB')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Temperature (TModB)')


    # RH vs GHI
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=data_frame['RH'], y=data_frame['GHI'])
    plt.title('Relative Humidity vs GHI')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Global Horizontal Irradiance (GHI)')

    # RH vs DNI
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=data_frame['RH'], y=data_frame['DNI'])
    plt.title('Relative Humidity vs DNI')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Direct Normal Irradiance (DNI)')

    plt.tight_layout()
    plt.show()

    # Correlation matrix for RH, Temperature, and Solar Radiation components
    rh_temp_radiation_corr_matrix = data_frame[['RH', 'TModA', 'TModB', 'GHI', 'DNI', 'DHI']].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(rh_temp_radiation_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap: RH, Temperature, and Solar Radiation Components')
    plt.show()

    # Trend Analysis with Line Plots
    plt.figure(figsize=(16, 10))

    # RH and TModA over time
    plt.subplot(2, 1, 1)
    plt.plot(data_frame['Timestamp'], data_frame['RH'], label='Relative Humidity (%)', color='blue', alpha=0.6)
    plt.plot(data_frame['Timestamp'], data_frame['TModA'], label='Temperature (TModA)', color='red', alpha=0.6)
    plt.legend(loc='upper right')
    plt.title('Relative Humidity and TModA Over Time')
    plt.xlabel('Time')
    plt.ylabel('Values')

    # RH and GHI over time
    plt.subplot(2, 1, 2)
    plt.plot(data_frame['Timestamp'], data_frame['RH'], label='Relative Humidity (%)', color='blue', alpha=0.6)
    plt.plot(data_frame['Timestamp'], data_frame['GHI'], label='Global Horizontal Irradiance (GHI)', color='green', alpha=0.6)
    plt.legend(loc='upper right')
    plt.title('Relative Humidity and GHI Over Time')
    plt.xlabel('Time')
    plt.ylabel('Values')

    plt.tight_layout()
    plt.show()

def plot_histograms(data):
    # List of variables to plot
    limited_data = data.head(1000)
    histogram_vars = ['GHI', 'DNI', 'DHI', 'WS', 'TModA', 'TModB']
    
    # Set the size of the overall figure
    plt.figure(figsize=(15, 10))

    # Loop through the variables and create a histogram for each
    for i, variable in enumerate(histogram_vars, start=1):
        plt.subplot(2, 3, i)
        sns.histplot(limited_data[variable], kde=True, bins=30, color='skyblue')
        plt.title(f'Histogram of {variable}')
        plt.xlabel(variable)
        plt.ylabel('Frequency')

    # Adjust the layout to avoid overlap
    plt.tight_layout()
    plt.show()

def calculate_z_scores(data, threshold=3):
    # List of variables to analyze
    limited_data = data.head(1000)
    z_score_vars = ['GHI', 'DNI', 'DHI', 'WS', 'TModA', 'TModB']
    
    # Create a dictionary to hold Z-scores
    z_scores_dict = {}

    # Calculate Z-scores for each variable
    for variable in z_score_vars:
        z_scores_dict[variable] = (limited_data[variable] - limited_data[variable].mean()) / limited_data[variable].std()

    # Convert the dictionary to a DataFrame for better handling
    z_scores_df = pd.DataFrame(z_scores_dict)
    
    # Flagging the data points with Z-scores beyond the threshold
    flagged_outliers = (z_scores_df.abs() > threshold)

    # Printing the flagged data points (outliers)
    for variable in z_score_vars:
        print(f"\nOutliers in {variable}:")
        print(limited_data[flagged_outliers[variable]])

    return z_scores_df, flagged_outliers

def plot_bubble_chart(data, x_var, y_var, size_var, title):
    limited_data = data.head(800)
    plt.figure(figsize=(12, 8))

    # Plotting the bubble chart
    plt.scatter(
        limited_data[x_var], 
        limited_data[y_var], 
        s=limited_data[size_var] * 10,
        alpha=0.5, 
        c=limited_data[size_var],  
        cmap='viridis', 
        edgecolors='w'
    )
    
    plt.title(title)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.colorbar(label=size_var)
    plt.show()


def clean_dataset(data):
    # Show initial details about missing entries
    print("Initial Missing Entries:")
    print(data.isnull().sum())


    # Remove columns that are fully null
    null_columns = data.columns[data.isnull().all()]
    data = data.drop(columns=null_columns)
    print(f"\nRemoved columns: {list(null_columns)}")

    # Eliminate rows with missing values in essential fields
    essential_fields = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']  # Adjust based on essential fields
    data = data.dropna(subset=essential_fields)
    print(f"\nRemoved rows missing essential fields. Remaining count: {len(data)}")

    # Handle anomalies by excluding outliers in specific fields
    for field in ['GHI', 'DNI', 'DHI', 'WS', 'TModA', 'TModB']:
        if field in data.columns:
            # Identify outliers using the Z-score approach
            z_scores = (data[field] - data[field].mean()) / data[field].std()
            data = data[(z_scores.abs() < 3)]  

    print("\nData Cleaning Finished")
    print("Dataset Details after Cleaning:")
    print(data.info())

    return data
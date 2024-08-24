import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

# Function to load data from a raw GitHub URL
def load_data_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Convert the response content into a pandas DataFrame
        data = pd.read_csv(StringIO(response.text))
        return data
    else:
        st.error(f"Failed to fetch data from GitHub. Status code: {response.status_code}")
        return pd.DataFrame()

# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    return outliers

def main():
    st.set_page_config(layout="wide")  # Set the page layout to wide

    st.title('MoonLight Energy Solutions')

    # Raw GitHub URLs
    data_files = {
        'Benin Malanville': 'https://raw.githubusercontent.com/Leulseged-Mesfin/Moonlight-Light-Energy-Solutions/main/Data/benin-malanville.csv',
        'Sierra Leone Bumbuna': 'https://raw.githubusercontent.com/Leulseged-Mesfin/Moonlight-Light-Energy-Solutions/main/Data/sierraleone-bumbuna.csv',
        'Togo Dapaong QC': 'https://raw.githubusercontent.com/Leulseged-Mesfin/Moonlight-Light-Energy-Solutions/main/Data/togo-dapaong_qc.csv'
    }

    # Dropdown for selecting data files
    selected_file = st.selectbox('Select a data file', list(data_files.keys()))

    # Construct the URL to fetch the selected data file
    data_url = data_files[selected_file]

    if selected_file:
        data = load_data_from_github(data_url)

        if not data.empty:
            # Dropdown for navigation options
            view_option = st.selectbox('Select view option', 
                                       ['Head', 'Tail', 'Summary Statistics', 'Data Quality Check'])

            # Display content based on selected view option
            if view_option in ['Head', 'Tail']:
                # Slider for selecting the number of rows to display
                num_rows = st.slider('Number of rows to display', min_value=1, max_value=min(len(data), 100), value=5)
                
                if view_option == 'Head':
                    # Show the first few rows of the dataframe
                    st.subheader('Data Head')
                    st.dataframe(data.head(num_rows))
                elif view_option == 'Tail':
                    # Show the last few rows of the dataframe
                    st.subheader('Data Tail')
                    st.dataframe(data.tail(num_rows))
            
            elif view_option == 'Summary Statistics':
                # Calculate and display summary statistics for numeric columns
                st.subheader('Summary Statistics for Numeric Columns')
                numeric_data = data.select_dtypes(include='number')  # Select numeric columns only
                
                if not numeric_data.empty:
                    # Calculate summary statistics
                    summary = pd.DataFrame()
                    summary['Mean'] = numeric_data.mean()
                    summary['Median'] = numeric_data.median()
                    summary['Standard Deviation'] = numeric_data.std()
                    summary['Variance'] = numeric_data.var()
                    summary['Range'] = numeric_data.max() - numeric_data.min()
                    summary['Skewness'] = numeric_data.skew()
                    summary['25th Percentile'] = numeric_data.quantile(0.25)
                    summary['50th Percentile (Median)'] = numeric_data.quantile(0.50)
                    summary['75th Percentile'] = numeric_data.quantile(0.75)

                    # Rename index for clarity
                    summary.index.name = 'Column'
                    st.dataframe(summary)
                else:
                    st.write('No numeric columns found in the data.')


        elif view_option == 'Data Quality Check':
                # Check for missing values, incorrect entries, and outliers
                st.subheader('Data Quality Check Results')

                # Create expandable sections for displaying results
                with st.expander("Missing Values"):
                    missing_values = data.isnull().sum()
                    st.write(missing_values[missing_values > 0])

                with st.expander("Incorrect Entries (Negative Values)"):
                    columns_positive = ['GHI', 'DNI', 'DHI']
                    incorrect_entries = {}
                    for col in columns_positive:
                        if col in data.columns:
                            incorrect_entries[col] = (data[col] < 0).sum()
                    st.write(pd.Series(incorrect_entries))

                with st.expander("Outliers Detected (IQR Method)"):
                    columns_outliers = ['ModA', 'ModB', 'WS', 'WSgust']
                    outlier_results = {}
                    for col in columns_outliers:
                        if col in data.columns:
                            outlier_results[col] = detect_outliers_iqr(data[col])
                    st.write(pd.Series(outlier_results))

    else:           
     st.write('Please select a data file to see the data.')


if __name__ == "__main__":
    main()
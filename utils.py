import pandas as pd
import datetime
import pandas as pd
import logging
import time


def keep_awake(iterable):
    for item in iterable:
        yield item
        time.sleep(0.1) 

def setup_logger():
    #TODO : ERASE FILE BEFORE
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("asset_management.log"),
                            logging.StreamHandler()
                        ])
    return logging.getLogger(__name__)

def load_data(data_file_path, factor_file_path, ret_var="stock_exret"):
    """
    Params:
    - data_file_path (str): Path to dataset
    - factor_file_path (str): Path to the predictor values
    - ret_var (str): The return variable (dependent variable)".
    
    Returns:
    - pd.DataFrame: The processed DataFrame with rank-transformed stock variables.
    """

# Log the start time
    # Turn off Pandas SettingWithCopyWarning
    pd.set_option("mode.chained_assignment", None)
    
    # Load the main stock data
    raw_data = pd.read_csv(data_file_path, parse_dates=["date"], low_memory=False)


    # Load the list of stock predictors (variables)
    stock_vars = list(pd.read_csv(factor_file_path)["variable"].values)
    
    print(stock_vars)

    filtered_data = raw_data[raw_data[ret_var].notna()].copy()
    filtered_data = raw_data[["date", "stock_exret"] + stock_vars].copy()  # Include the 'date' and 'stock_exret' columns

    # Group data by date (monthly)
    monthly_groups = filtered_data.groupby("date")
    
    # Initialize an empty DataFrame to store processed data
    processed_data = pd.DataFrame()
    
    # Process each group (month) separately
    for date, monthly_data in monthly_groups:
        group = monthly_data.copy()
        
        # Fill missing values with the median for each variable
        for var in stock_vars:
            var_median = group[var].median(skipna=True)
            group[var] = group[var].fillna(var_median)
        
        # Rank transform each variable to [-1, 1]
        for var in stock_vars:
            group[var] = group[var].rank(method="dense") - 1
            group_max = group[var].max()
            group[var] = (group[var] / group_max) * 2 - 1
        
        # Append processed group to the final DataFrame
        processed_data = processed_data._append(group, ignore_index=True)
    #     # print(processed_data)

    
    
    # Return the processed DataFrame
    return filtered_data
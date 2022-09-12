import os
from pathlib import Path
import pandas as pd

#########################   DATA LOADING/MANIPULATION PRIMITIVES   #########################

def get_data(csv_name: str, id: str, label: str, verbose=False) -> pd.DataFrame:
    # Build data path
    f_path = os.path.join(str(Path(__file__).parent.absolute()), 'data', csv_name)
    # Read csv
    df = pd.read_csv(f_path, index_col=id, skipinitialspace=True)

    # Build the reduced dataset
    label_df = df.loc[:, label]
    df = df.drop(label, axis=1)
    final_df = df.loc[:, :'HDDparticipant'].join(df.loc[:, 'ColumnID':'OtherP']).join(label_df)

    if verbose: print(final_df.head())

    return final_df

def get_selected_data(csv_name: str, id: str, label: str, verbose=False) -> pd.DataFrame:
    # Build data path
    f_path = os.path.join(str(Path(__file__).parent.absolute()), 'data', csv_name)
    # Read csv
    df = pd.read_csv(f_path, index_col=id, skipinitialspace=True)

    # Build the reduced dataset
    label_df = df.loc[:, label]
    df = df.drop(label, axis=1)
    final_df = df.loc[:, 'TTRparticipant':'HDDparticipant'].join(label_df)

    if verbose: print(final_df.head())

    return final_df

def remove_nan(df: pd.DataFrame) -> pd.DataFrame:
    # Define the list of rows that contains any NaN
    to_remove = df[df.notnull().sum(axis=1)<len(df.columns)].index.tolist()
    # Remove the rows from the dataset
    final_df = df.drop(to_remove, axis=0)

    return final_df

def remove_useless_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(['ColumnID', 'Segment'], axis=1)

def reorg_to_visualize(full_data):
    data_names, num_data = zip(*full_data)
    rmse_avgs, rmse_stdevs, mae_avgs, mae_stdevs, r2_avgs, r2_stdevs = [], [], [], [], [], []
    for res in num_data:
        rmse_avg, rmse_stdev, mae_avg, mae_stdev, r2_avg, r2_stdev = zip(*res)
        rmse_avgs.append(rmse_avg), rmse_stdevs.append(rmse_stdev)
        mae_avgs.append(mae_avg), mae_stdevs.append(mae_stdev)
        r2_avgs.append(r2_avg), r2_stdevs.append(r2_stdev)

    return data_names, (rmse_avgs, rmse_stdevs), (mae_avgs, mae_stdevs), (r2_avgs, r2_stdevs)
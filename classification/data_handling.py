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
    data_names, num_data, _ = zip(*full_data)
    acc_avgs, acc_stdevs    = [], []
    prec_avgs, prec_stdevs  = [], []
    rec_avgs, rec_stdevs    = [], []
    f1_avgs, f1_stdevs      = [], []
    for res in num_data:
        acc_avg, acc_stdev, prec_avg, prec_stdev, rec_avg, rec_stdev, f1_avg, f1_stdev = zip(*res)
        acc_avgs.append(acc_avg), acc_stdevs.append(acc_stdev)
        prec_avgs.append(prec_avg), prec_stdevs.append(prec_stdev)
        rec_avgs.append(rec_avg), rec_stdevs.append(rec_stdev)
        f1_avgs.append(f1_avg), f1_stdevs.append(f1_stdev)

    return data_names, (acc_avgs, acc_stdevs), (prec_avgs, prec_stdevs), (rec_avgs, rec_stdevs), (f1_avgs, f1_stdevs)
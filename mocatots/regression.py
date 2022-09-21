import os
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
import numpy as np
import smogn

import plots, data_handling

# Data Manipulation
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Sklearn models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Needed to avoid warning errors
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



#########################   HIGH-LEVEL PROCESSING PHASES   #########################

def load_dataset(verbose = False) -> pd.DataFrame:
    # Load the full dataset
    full_data_df = data_handling.get_data('partialData.csv', 'REGTRYID', 'MOCATOTS', verbose=verbose)
    # Clean the dataset fixing NaN entries
    full_data_df = data_handling.remove_nan(full_data_df)
    # Convert string categorical value to numerical range
    full_data_df = data_handling.remove_useless_cols(full_data_df)

    return full_data_df

def load_selected_dataset() -> pd.DataFrame:
    # Load the selected dataset
    selected_data_df = data_handling.get_selected_data('partialData.csv', 'REGTRYID', 'MOCATOTS')
    # Clean the dataset fixing NaN entries
    selected_data_df = data_handling.remove_nan(selected_data_df)

    return selected_data_df

def initialize(df: pd.DataFrame, selected_df: pd.DataFrame, num_folds: int) -> np.ndarray:
    # Split data from labels
    full_data, selected_data = df.to_numpy(), selected_df.to_numpy()
    X, y = full_data[:, :-1], full_data[:, -1]
    X_sel, y_sel = selected_data[:, :-1], selected_data[:, -1]

    # Create the reduced datasets
    pca32, pca64 = PCA(n_components=32), PCA(n_components=64)
    X_pca32, X_pca64 = pca32.fit_transform(X), pca64.fit_transform(X)

    # Create train/test division 5 time to average results at the end
    # Unfortunately, we do not have enough data to stratify over the label
    repeated_train_full, repeated_test_full = [], []
    repeated_train_pca32, repeated_test_pca32 = [], []
    repeated_train_pca64, repeated_test_pca64 = [], []
    repeated_train_sel, repeated_test_sel = [], []
    for seed in range(num_folds):
        # Define full plit train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
        repeated_train_full.append((X_train, y_train)), repeated_test_full.append((X_test, y_test))

        # PCA with 32 dimensions
        X_train, X_test, y_train, y_test = train_test_split(X_pca32, y, test_size=0.25, random_state=seed)
        repeated_train_pca32.append((X_train, y_train)), repeated_test_pca32.append((X_test, y_test))

        # PCA with 64 dimensions
        X_train, X_test, y_train, y_test = train_test_split(X_pca64, y, test_size=0.25, random_state=seed)
        repeated_train_pca64.append((X_train, y_train)), repeated_test_pca64.append((X_test, y_test))

        # Define selected plit train/test
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y_sel, test_size=0.25, random_state=seed)
        repeated_train_sel.append((X_train, y_train)), repeated_test_sel.append((X_test, y_test))


    return (repeated_train_full, repeated_test_full), (repeated_train_pca32, repeated_test_pca32), \
           (repeated_train_pca64, repeated_test_pca64), (repeated_train_sel, repeated_test_sel)

def train(train_data: list, constructor, parameters: dict = None) -> list:
    models = []
    for X, y in train_data:
        model = constructor()
        if parameters:
            model.set_params(**parameters)

        model = model.fit(X, y)
        models.append(model)
    
    return models

def test(test_data: list, trained_models: list) -> list:
    results = []
    for (X, y_true), model in zip(test_data, trained_models):
        y_pred = model.predict(X)
        results.append((y_true, y_pred))

    return results

def evaluate(results: list):
    rmse, mae, r2 = [], [], []
    for y_true, y_pred in results:
        rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae.append(mean_absolute_error(y_true, y_pred))
        r2.append(r2_score(y_true, y_pred))

    print(f'   RMSE ->   Mean: {np.mean(rmse):.4f}\tStdev: {np.std(rmse):.4f}\tSkew: {skew(rmse):.4f}\tKurtosis: {kurtosis(rmse):.4f}')
    print(f'   MAE  ->   Mean: {np.mean(mae):.4f}\tStdev: {np.std(mae):.4f}\tSkew: {skew(mae):.4f}\tKurtosis: {kurtosis(mae):.4f}')
    print(f'   R2   ->   Mean: {np.mean(r2):.4f}\tStdev: {np.std(r2):.4f}\tSkew: {skew(r2):.4f}\tKurtosis: {kurtosis(r2):.4f}')

    return (np.mean(rmse), np.std(rmse), skew(rmse), kurtosis(rmse)),\
           (np.mean(mae), np.std(mae), skew(mae), kurtosis(mae)),\
           (np.mean(r2), np.std(r2), skew(r2), kurtosis(r2))

def save_extended_results(results: list, dataset: str) -> None:
    with open(os.path.join('results', f'{dataset}.csv'), 'w+') as f:
        # First write the columns of the csv
        f.write('Model,RMSE_mean,RMSE_stdev,RMSE_skew,RMSE_kurtosis,MAE_mean,MAE_stdev,MAE_skew,MAE_kurtosis,R2_mean,R2_stdev,R2_skew,R2_kurtosis\n')
        # Write all the extended results
        for model, (rmse, mae, r2) in results:
            line = model
            line += f',{rmse[0]:.4f},{rmse[1]:.4f},{rmse[2]:.4f},{rmse[3]:.4f}'
            line += f',{mae[0]:.4f},{mae[1]:.4f},{mae[2]:.4f},{mae[3]:.4f}'
            line += f',{r2[0]:.4f},{r2[1]:.4f},{r2[2]:.4f},{r2[3]:.4f}'
            f.write(line + '\n')

    
def visualize(full_data: list, full_df: pd.DataFrame):
    # Change shape of the data
    data_names, rmse, mae, r2 = data_handling.reorg_to_visualize(full_data)

    # Plot performance
    for data, metric in zip([rmse, mae, r2], ['RMSE', 'MAE', 'R2']):
        plots.plot_performance(data_names, data[0], data[1], ['Ridge Regression', 'Random Forest', 'Gradient Boosting'], metric, 0.2)

    # Plot score distribution
    plots.plot_score_distrib(full_df)


#########################   MAIN FUNCTION   #########################

def main():
    NUM_FOLDS = 10
    SMOTE = False

    # Firt, simply load the datasets
    if SMOTE:
        full_data_df = smogn.smoter(data = load_dataset().reset_index(), y = 'MOCATOTS')
        selected_data_df = smogn.smoter(data = load_selected_dataset().reset_index(), y = 'MOCATOTS')
    else:
        full_data_df = load_dataset()
        selected_data_df = load_selected_dataset()

    # Initialize the dataframe and prepare for training/testing
    full_data, pca32_data, pca64_data, selected_data = initialize(full_data_df, selected_data_df, NUM_FOLDS)

    # Define the dataset and models lists with associated names
    datasets = [(full_data, 'Full Dataset'), (pca32_data, '32 Components Dataset'), (pca64_data, '64 Components Dataset'), (selected_data, 'Feature Selected Dataset')]
    model_constructors = [(Ridge, 'Ridge Regression'), (RandomForestRegressor, 'Random Forest'), (GradientBoostingRegressor, 'Gradient Boosting')]
    # Define Hyperparameters for the models
    hyperparameters = [{'normalize': True},
                       {'max_depth': 1},
                       {'loss': 'absolute_error', 'learning_rate': 0.01}]
    # Train and test for all the models
    data_res = []
    for data, data_name in datasets:
        print('\n' +'#'*90)
        print(f'Using {data_name}')
        final_res, short_res = [], []
        for (constructor, model_name), params in zip(model_constructors, hyperparameters):
            print(f'\nRoutine started for {model_name}')
            # Train the models
            models = train(data[0], constructor, params)

            # Test the models
            results = test(data[1], models)

            # Evaluate results
            rmse, mae, r2 = evaluate(results)
            final_res.append((model_name, (rmse, mae, r2)))
            short_res.append((rmse[0], rmse[1], mae[0], mae[1], r2[0], r2[1]))

        # Save all the metrics computed in a dedicated file
        save_extended_results(final_res, data_name)
        # Record needed data to lated do some visualization
        data_res.append((data_name, short_res))
        print('#'*90)
    
    # Visualize anything interesting
    visualize(data_res, full_data_df)

if __name__ == '__main__':
    main()
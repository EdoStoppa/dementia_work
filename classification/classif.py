import os
import pandas as pd
import numpy as np
import plots, data_handling

# Data Manipulation
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Sklearn models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Metrics
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Needed to avoid warning errors
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



#########################   HIGH-LEVEL PROCESSING PHASES   #########################

def load_dataset(label: str, verbose = False) -> pd.DataFrame:
    # Load the full dataset
    full_data_df = data_handling.get_data('fullData.csv', 'REGTRYID', label, verbose=verbose)
    # Clean the dataset fixing NaN entries
    full_data_df = data_handling.remove_nan(full_data_df)
    # Convert string categorical value to numerical range
    full_data_df = data_handling.remove_useless_cols(full_data_df)

    return full_data_df

def load_selected_dataset(label: str) -> pd.DataFrame:
    # Load the selected dataset
    selected_data_df = data_handling.get_selected_data('fullData.csv', 'REGTRYID', label)
    # Clean the dataset fixing NaN entries
    selected_data_df = data_handling.remove_nan(selected_data_df)

    return selected_data_df

def initialize(df: pd.DataFrame, selected_df: pd.DataFrame, num_folds: int, smote: bool, pca_dims) -> np.ndarray:
    # Split data from labels
    full_data, selected_data = df.to_numpy(), selected_df.to_numpy()
    X, y = full_data[:, :-1], full_data[:, -1]
    X_sel, y_sel = selected_data[:, :-1], selected_data[:, -1]

    if smote:
        smt = SMOTE()
        X, y = smt.fit_resample(X, y)
        X_sel, y_sel = smt.fit_resample(X_sel, y_sel)

    # Create the reduced datasets
    pca1, pca2 = PCA(n_components=pca_dims[0]), PCA(n_components=pca_dims[1])
    X_pca1, X_pca2 = pca1.fit_transform(X), pca2.fit_transform(X)

    # Create train/test division 5 time to average results at the end
    # Unfortunately, we do not have enough data to stratify over the label
    repeated_train_full, repeated_test_full = [], []
    repeated_train_pca1, repeated_test_pca1 = [], []
    repeated_train_pca2, repeated_test_pca2 = [], []
    repeated_train_sel, repeated_test_sel = [], []
    for seed in range(num_folds):
        # Define full plit train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=(y if smote else None))
        repeated_train_full.append((X_train, y_train)), repeated_test_full.append((X_test, y_test))

        # First PCA
        X_train, X_test, y_train, y_test = train_test_split(X_pca1, y, test_size=0.25, random_state=seed, stratify=(y if smote else None))
        repeated_train_pca1.append((X_train, y_train)), repeated_test_pca1.append((X_test, y_test))

        # Second PCA
        X_train, X_test, y_train, y_test = train_test_split(X_pca2, y, test_size=0.25, random_state=seed, stratify=(y if smote else None))
        repeated_train_pca2.append((X_train, y_train)), repeated_test_pca2.append((X_test, y_test))

        # Define selected plit train/test
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y_sel, test_size=0.25, random_state=seed, stratify=(y_sel if smote else None))
        repeated_train_sel.append((X_train, y_train)), repeated_test_sel.append((X_test, y_test))


    return (repeated_train_full, repeated_test_full), (repeated_train_pca1, repeated_test_pca1), \
           (repeated_train_pca2, repeated_test_pca2), (repeated_train_sel, repeated_test_sel)

def train(train_data: list, constructor, parameters: dict = None) -> list:
    models = []
    for X, y in train_data:
        # Instantiate the model
        model = constructor()
        if parameters:
            # Set hyperparameters
            model.set_params(**parameters)
        # Train the model
        model = model.fit(X, y)
        # Save the model
        models.append(model)
    
    return models

def test(test_data: list, trained_models: list) -> list:
    results = []
    for (X, y_true), model in zip(test_data, trained_models):
        # Test the model
        y_pred = model.predict(X)
        # Obtain prediction probabilities (used for ROC curve)
        y_prob = model.predict_proba(X)[:, 1]
        # Save results
        results.append((y_true, y_pred, y_prob))

    return results

def evaluate(results: list):
    acc, prec, rec, f1 = [], [], [], []
    for y_true, y_pred, _ in results:
        acc.append(accuracy_score(y_true, y_pred))
        prec.append(precision_score(y_true, y_pred))
        rec.append(recall_score(y_true, y_pred))
        f1.append(f1_score(y_true, y_pred))

    print(f'   Accuracy   ->   Mean: {np.mean(acc):.4f}\tStdev: {np.std(acc):.4f}\tSkew: {skew(acc):.4f}\tKurtosis: {kurtosis(acc):.4f}')
    print(f'   Precision  ->   Mean: {np.mean(prec):.4f}\tStdev: {np.std(prec):.4f}\tSkew: {skew(prec):.4f}\tKurtosis: {kurtosis(prec):.4f}')
    print(f'   Recall     ->   Mean: {np.mean(rec):.4f}\tStdev: {np.std(rec):.4f}\tSkew: {skew(rec):.4f}\tKurtosis: {kurtosis(rec):.4f}')
    print(f'   F1         ->   Mean: {np.mean(f1):.4f}\tStdev: {np.std(f1):.4f}\tSkew: {skew(f1):.4f}\tKurtosis: {kurtosis(f1):.4f}')

    return (np.mean(acc), np.std(acc), skew(acc), kurtosis(acc)),\
           (np.mean(prec), np.std(prec), skew(prec), kurtosis(prec)),\
           (np.mean(rec), np.std(rec), skew(rec), kurtosis(rec)),\
           (np.mean(f1), np.std(f1), skew(f1), kurtosis(f1))

def save_extended_results(results: list, dataset: str, label: str, smote: bool) -> None:
    with open(os.path.join('results', 'extended', f'{"" if smote else "no_"}smote', label, f'{dataset}.csv'), 'w+') as f:
        # First write the columns of the csv
        f.write('Model,Accuracy_mean,Accuracy_stdev,Accuracy_skew,Accuracy_kurtosis,' + 
                'Precision_mean,Precision_stdev,Precision_skew,Precision_kurtosis,' +
                'Recall_mean,Recall_stdev,Recall_skew,Recall_kurtosis,' + 
                'F1_mean,F1_stdev,F1_skew,F1_kurtosis\n')
        # Write all the extended results
        for model, (acc, prec, rec, f1) in results:
            line = model
            line += f',{acc[0]:.4f},{acc[1]:.4f},{acc[2]:.4f},{acc[3]:.4f}'
            line += f',{prec[0]:.4f},{prec[1]:.4f},{prec[2]:.4f},{prec[3]:.4f}'
            line += f',{rec[0]:.4f},{rec[1]:.4f},{rec[2]:.4f},{rec[3]:.4f}'
            line += f',{f1[0]:.4f},{f1[1]:.4f},{f1[2]:.4f},{f1[3]:.4f}'
            f.write(line + '\n')

    
def visualize(full_data: list, full_df: pd.DataFrame, label:str, smote: bool):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    algos = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    # Change shape of the data
    data_names, acc, prec, rec, f1 = data_handling.reorg_to_visualize(full_data)

    # Plot performance
    for data, metric in zip([acc, prec, rec, f1], metrics):
        plots.plot_performance(data_names, data[0], data[1], algos, metric, 0.1, label, smote)

    for data_name, _, roc_data in full_data:
        plots.plot_roc_curve(data_name, roc_data, algos, label, smote)

    # Plot score distribution
    plots.plot_distrib(full_df, label)


#########################   MAIN FUNCTION   #########################

def main(label: str, num_folds: int, smote: bool, pca_dims=(32, 64)):

    full_data_df = load_dataset(label)
    selected_data_df = load_selected_dataset(label)

    # Initialize the dataframe and prepare for training/testing
    full_data, pca_data1, pca_data2, selected_data = initialize(full_data_df, selected_data_df, num_folds, smote, pca_dims)

    # Define the dataset and models lists with associated names
    datasets = [(full_data, 'Full'), (pca_data1, f'{pca_dims[0]} Components'), (pca_data2, f'{pca_dims[1]} Components'), (selected_data, 'Feature Selected')]
    model_constructors = [(LogisticRegression, 'Logistic Regression'), (RandomForestClassifier, 'Random Forest'), (GradientBoostingClassifier, 'Gradient Boosting')]
    # Define Hyperparameters for the models
    hyperparameters = [{'max_iter': 10000},
                       {'n_estimators': 200},
                       {'learning_rate': 0.1}]
    # Train and test for all the models
    data_res = []
    for (train_data, test_data), data_name in datasets:
        print('\n' +'#'*90)
        print(f'Using {data_name}')
        final_res, short_res, prob_res = [], [], []
        for (constructor, model_name), params in zip(model_constructors, hyperparameters):
            print(f'\nRoutine started for {model_name}')
            # Train the models
            models = train(train_data, constructor, params)

            # Test the models
            results = test(test_data, models)

            # Evaluate results
            acc, prec, rec, f1 = evaluate(results)
            final_res.append((model_name, (acc, prec, rec, f1)))
            short_res.append((acc[0], acc[1], prec[0], prec[1], rec[0], rec[1], f1[0], f1[1]))
            prob_res.append((results[0][0], results[0][2]))

        # Save all the metrics computed in a dedicated file
        save_extended_results(final_res, data_name, label, smote)
        # Record needed data to lated do some visualization
        data_res.append((data_name, short_res, prob_res))
        print('#'*90)
    
    # Visualize anything interesting
    visualize(data_res, full_data_df, label, smote)

if __name__ == '__main__':
    # MOCA_impairment
    main('MOCA_impairment', num_folds=10, smote=False)
    main('MOCA_impairment', num_folds=10, smote=True)
    # tTau_AB42Positivity
    main('tTau_AB42Positivity', num_folds=10, smote=False, pca_dims=(16,32))
    main('tTau_AB42Positivity', num_folds=10, smote=True, pca_dims=(16,32))
    # AB42_AB40Positivity
    main('AB42_AB40Positivity', num_folds=10, smote=False, pca_dims=(16,32))
    main('AB42_AB40Positivity', num_folds=10, smote=True, pca_dims=(16,32))

    
import numpy as np
from plots import plot_best_feat
from classif import load_dataset
# Data and models manipulation
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
# Imbalanced dataset fix
from imblearn.over_sampling import SMOTE
# Others
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)



def get_imp_lasso(pipeline: Pipeline, X, y):
    # Define a GridSearch for Lasso Regression
    search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1, 10, 0.1)},
                      cv = 3,
                      scoring="neg_mean_squared_error")
    # Find best Hypeparameters
    search.fit(X, y)
    # Obtain the Lasso coefficients
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)

    return importance

def get_imp_tree(pipeline, X, y):
    # Define a GridSearch for Decision Tree
    search = GridSearchCV(pipeline,
                      {'model__max_depth': np.arange(1, 15, 1)},
                      cv = 3,
                      scoring = "neg_mean_squared_error")
    # Find best Hypeparameters
    search.fit(X, y)
    # Obtain the Decision Tree feature importances
    importance = search.best_estimator_.named_steps['model'].feature_importances_

    return importance


def best_feat(func, pipeline, X, y, runs=10):
    best = None
    for _ in range(runs):
        # Get the train/test split
        X_train, _, y_train, _ = train_test_split(X, y, train_size=0.2)
        # Run the function realted to the model used (ex: get_imp_tree)
        imp = func(pipeline, X_train, y_train)
        # Update the best features
        if best is not None:
            best += imp
        else:
            best = imp

    return imp

def get_topK(X, y, cols, score_func, runs=10, k=16):
    # Create object that will rank features based on the score function
    select = SelectKBest(score_func=score_func)
    top_ks = []
    top_scores = []
    for _ in range(runs):
        # Obtain the scores for all the features
        scores = list(select.fit(X, y).scores_)
        # Associate each score to the feature that is related to
        n_scores = list(zip(scores, [i for i in range(len(scores))]))
        # Sort the scores from highest to lowest
        n_scores.sort(key=lambda x: x[0], reverse=True)
        # Retrieve scores and features indexes
        scores, pos = zip(*n_scores)
        # Save top-k scores and their feature names
        top_scores.append(list(scores[:k]))
        top_ks.append(list(cols[list(pos)[:k]]))

    return top_ks, top_scores
    
    
def stat_analysis(LABEL: str):
    # Load the dataset
    df = load_dataset(LABEL)
    # Get the column names (without the label)
    cols = df.drop([LABEL], axis=1).columns.to_numpy()
    # Separate features from labels
    full_data = df.to_numpy()
    X, y = full_data[:, :-1], full_data[:, -1]
    # Use SMOTE
    smt = SMOTE()
    X, y = smt.fit_resample(X, y)

    # F Function
    f_top_ks, f_top_scores = get_topK(X, y, cols, f_classif, runs=1)
    f_top_ks, f_top_scores = f_top_ks[0], f_top_scores[0]
    plot_best_feat(f_top_ks, f_top_scores, 'Scores', 'F_Function', LABEL, space=2)
    
    # Mutual Information
    mi_top_ks, mi_top_scores = get_topK(X, y, cols, mutual_info_classif, runs=10)
    # Get the cumulative scores divided by the number of runs
    mi_tot_scores = {}
    for ks, tops in zip(mi_top_ks, mi_top_scores):
        for name, score in zip(ks, tops):
            if name in mi_tot_scores: mi_tot_scores[name] += score/10
            else:                     mi_tot_scores[name]  = score/10
    # Get the bin counts for each feature present
    mi_finals = []
    for l in mi_top_ks: mi_finals += l
    unique, count = np.unique(mi_finals, return_counts=True)
    mi_bins = list(zip(list(unique), list(count)))
    mi_bins.sort(key=lambda x: x[1], reverse=True)
    unique, count = zip(*mi_bins)
    plot_best_feat(list(unique), list(count), 'Frequency', 'Mutual_Info', LABEL, space=2)
    plot_best_feat(list(unique), [mi_tot_scores[k] for k in unique], 'Scores', 'Mutual_Info', LABEL, space=2)
    

def ml(LABEL: str):
    # Load the dataset
    df = load_dataset(LABEL)
    # Get the column names (without the label)
    cols = df.drop([LABEL], axis=1).columns.to_numpy()
    # Separate features from labels
    full_data = df.to_numpy()
    X, y = full_data[:, :-1], full_data[:, -1]
    # Use SMOTE
    smt = SMOTE()
    X, y = smt.fit_resample(X, y)

    # LASSO
    lasso_pip = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', Lasso())])
    lasso_imp = best_feat(get_imp_lasso, lasso_pip, X, y, runs=50)
    print(lasso_imp[lasso_imp > 0])
    print(cols[lasso_imp > 0], '\n')

    # DECISION TREE
    tree_pip = Pipeline([
                     ('scaler', StandardScaler()),
                     ('model', DecisionTreeClassifier())])
    tree_imp = best_feat(get_imp_tree, tree_pip, X, y, runs=100)
    print(tree_imp[tree_imp > 0])
    print(cols[tree_imp > 0], '\n')


if __name__ == '__main__':
    stat_analysis('MOCA_impairment')
    stat_analysis('AB42_AB40Positivity')
    stat_analysis('tTau_AB42Positivity')
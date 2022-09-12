import numpy as np
from plots import plot_best_feat
from regression import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_imp_lasso(pipeline: Pipeline, X, y):

    search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1, 10, 0.1)},
                      cv = 3,
                      scoring="neg_mean_squared_error")

    search.fit(X, y)
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)

    return importance

def get_imp_tree(pipeline, X, y):
    search = GridSearchCV(pipeline,
                      {'model__max_depth': np.arange(1, 15, 1)},
                      cv = 3,
                      scoring = "neg_mean_squared_error")
    search.fit(X, y)
    importance = search.best_estimator_.named_steps['model'].feature_importances_

    return importance


def best_feat(func, pipeline, X, y, runs=10):
    best = None
    for _ in range(runs):
        X_train, _, y_train, _ = train_test_split(X, y, train_size=0.2)
        imp = func(pipeline, X_train, y_train)
        #imp = func(pipeline, X, y)

        if best is not None:
            best += imp
        else:
            best = imp

    return imp

def get_topK(X, y, cols, func, runs=10, k=16):
    select = SelectKBest(score_func=func)
    top_ks = []
    top_scores = []
    for _ in range(runs):
        scores = list(select.fit(X, y).scores_)
        n_scores = list(zip(scores, [i for i in range(len(scores))]))
        n_scores.sort(key=lambda x: x[0], reverse=True)
        scores, pos = zip(*n_scores)

        top_scores.append(list(scores[:k]))
        top_ks.append(list(cols[list(pos)[:k]]))

    return top_ks, top_scores
    
    
def stat_analysis():
    df = load_dataset()
    cols = df.drop(['MOCATOTS'], axis=1).columns.to_numpy()
    full_data = df.to_numpy()
    X, y = full_data[:, :-1], full_data[:, -1]

    # F Function
    f_top_ks, f_top_scores = get_topK(X, y, cols, f_regression, runs=1)
    f_top_ks, f_top_scores = f_top_ks[0], f_top_scores[0]
    plot_best_feat(f_top_ks, f_top_scores, 'Scores', 'F_Function', space=2)
    
    # Mutual Information
    mi_top_ks, mi_top_scores = get_topK(X, y, cols, mutual_info_regression, runs=10)
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
    plot_best_feat(list(unique), list(count), 'Frequency', 'Mutual_Info', space=2)
    plot_best_feat(list(unique), [mi_tot_scores[k] for k in unique], 'Scores', 'Mutual_Info', space=2)
    

def ml():
    print()
    df = load_dataset()
    cols = df.drop(['MOCATOTS'], axis=1).columns.to_numpy()
    full_data = df.to_numpy()
    X, y = full_data[:, :-1], full_data[:, -1]

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
                     ('model', DecisionTreeRegressor())])
    tree_imp = best_feat(get_imp_tree, tree_pip, X, y, runs=100)
    print(tree_imp[tree_imp > 0])
    print(cols[tree_imp > 0], '\n')


    

stat_analysis()
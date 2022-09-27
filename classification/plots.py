import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

#########################   PLOT FUNCTIONS   #########################
def plot_distrib(data: pd.DataFrame, label: str) -> None:
    df = data.value_counts(label).sort_index()

    plt.title(f'{label} distribution')
    plt.pie(df.values, labels=[int(e) for e in df.index.to_list()], autopct='%1.0f%%')
    plt.savefig(os.path.join('results', 'imgs', 'classes_distribution', f'{label}.png'))
    plt.clf()
    plt.cla()

def plot_performance(data_names: list, avgs: list, stdevs: list, algos: list, metric: str, precision: float, label: str, smote: bool):    
    # Extract the max value on the Y axis
    max_avg_y = max([max(item) for item in avgs])
    max_stdev_y = max([max(item) for item in stdevs])
    max_y = np.ceil(max_avg_y) if max_avg_y + max_stdev_y < np.ceil(max_avg_y) else np.ceil(max_avg_y + max_stdev_y)

    min_avg_y = min([min(item) for item in avgs])
    min_stdev_y = min([max(item) for item in stdevs])
    min_y = np.floor(min_avg_y) if min_avg_y - min_stdev_y < np.floor(min_avg_y) else np.floor(min_avg_y - min_stdev_y)

    # Rearrange the data in order to be in the correct form to plot everything
    avgs = list(map(list, zip(*avgs)))
    stdevs = list(map(list, zip(*stdevs)))

    # Create positions of the bars
    w = 0.12
    xticks = [np.arange(len(avgs[0]))]
    for _ in range(1, len(avgs)):
        xticks.append(xticks[-1] + w)

    # Actually plot everything
    for values, stdev, pos, lbl in zip(avgs, stdevs, xticks, algos):
        plt.bar(pos, values, w, label=lbl)
        plt.errorbar(pos, values, yerr=stdev, fmt="o", ecolor='black', capsize=3, color='black')

    # Choose the right position to put the labels on
    num_datasets = len(avgs)
    if num_datasets % 2 == 0:
        idx = num_datasets//2 - 1
        pos = xticks[idx] + w/2
    else:
        idx = (num_datasets - 1) // 2
        pos = xticks[idx]
    
    plt.title(f'Performance ({label})')
    plt.ylabel(metric)
    plt.xlabel('Dataset')
    plt.xticks(pos, data_names, rotation = 20)
    plt.yticks(np.arange(min(min_y, 0), min(max_y, 1.1), precision))
    # Finish everything and show it
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'imgs', f'performance_{"" if smote else "no_"}smote', label, f'{metric.lower()}.png'))
    plt.clf()
    plt.cla()

def plot_roc_curve(data_name: list, data: list, algos: list, label: str, smote: bool):
    for (y_true, y_prob), algo in zip(data, algos):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, marker='.', label=algo)
    plt.title(f'{label} Roc curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'imgs', f'performance_{"" if smote else "no_"}smote', label, 'ROC Curves', f'{data_name}.png'))
    plt.clf()
    plt.cla()

def plot_best_feat(feats, scores, type_plot: str, func: str, lbl: str, space=1):
    pos = np.arange(len(feats))*space
    plt.bar(pos, scores)
    plt.title(f'Best Features {type_plot} ({func})')
    plt.ylabel(f'{type_plot}')
    plt.xlabel('Features')
    plt.xticks(pos, feats, rotation = 90)
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'imgs', 'best_features', f'{lbl}', f'{func.lower()}_{type_plot.lower()}.png'))
    plt.clf()
    plt.cla()
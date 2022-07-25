import numpy as np
from sklearn.model_selection import KFold

from baseline_model.data_loader import load_data

from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

from baseline_model.feature_creator import sent2features

import matplotlib.pyplot as plt


def main():
    #dataset_path = "C:/Dev/Smart_Data/E4/datasets/output.xlsx"
    dataset_path = "C:/Dev/Smart_Data/E4/datasets/crest_1_2_13.xlsx"
    number_splits = 5
    seeds = [0, 1, 42, 47, 101]

    x_data, y_data = load_data(dataset_path)
    x_data = [sent2features(s) for s in x_data]

    x_data = np.asarray(x_data, dtype=object)
    y_data = np.asarray(y_data, dtype=object)

    kf = KFold(n_splits=number_splits, shuffle=True)

    y_test_arr = np.zeros((len(seeds), number_splits), dtype=object)
    y_pred_arr = np.zeros((len(seeds), number_splits), dtype=object)

    for seed_id, seed in enumerate(seeds):
        np.random.seed(seed)
        print(f"CV {seed_id}:")
        for run, (train_index, test_index) in enumerate(kf.split(x_data)):
            print(f"    Run {run}")
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False)
            crf.fit(x_train, y_train)

            y_pred = crf.predict(x_test)

            y_test_arr[seed_id][run] = list(y_test)
            y_pred_arr[seed_id][run] = list(y_pred)


    print("ALL: ")

    plot_results(y_test_arr, y_pred_arr)


def plot_results(y_test_arr, y_pred_arr):
    tags = ["C", "E", "O"]
    seeds = [0, 1, 42, 47, 101]
    colors = ["blue", "red", "yellow", "green", "orange"]

    shape = y_pred_arr.shape
    num_seeds = len(seeds)
    num_splits = shape[1]
    print(shape)
    f1_scores = np.zeros((num_seeds, num_splits, 3))

    for seed_id in range(num_seeds):
        for split_id in range(num_splits):
            y_test = y_test_arr[seed_id, split_id]
            y_pred = y_pred_arr[seed_id, split_id]
            report = metrics.flat_classification_report(y_test, y_pred, digits=3, output_dict=True)
            for i, tag in enumerate(tags):
                f1_scores[seed_id, split_id, i] = report[tag]["f1-score"]

    means = f1_scores.mean(axis=1)
    stds = f1_scores.std(axis=1)
    xs = np.asarray([0.0, 1.0, 2.0])
    plt.ylim([0, 1.0])
    #for x, tag in enumerate(tags):
    for s, seed in enumerate(seeds):
        color = colors[s]
        plt.bar(
            xs + 0.1 * s - 0.2,
            means[s],
            yerr=stds[s],
            width=0.1,
            color=color,
            edgecolor="black",
            align="center",
            label=str(seed)
        )
    plt.ylabel("F1-Score")
    plt.xlabel("Tags")
    plt.legend(title="Seeds")
    plt.xticks([0,1,2],["Cause", "Effect", "Other"])
    plt.show()


if __name__ == "__main__":
    main()
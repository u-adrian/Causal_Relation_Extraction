import numpy as np
from sklearn.model_selection import KFold

from baseline_model.data_loader import load_data

from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

from baseline_model.feature_creator import sent2features


def main():
    dataset_path = "C:/Dev/Smart_Data/E4/datasets/output.xlsx"
    number_splits = 5
    seeds = [0,1,42,101]

    x_data, y_data = load_data(dataset_path)
    x_data = [sent2features(s) for s in x_data]

    x_data = np.asarray(x_data, dtype=object)
    y_data = np.asarray(y_data, dtype=object)

    kf = KFold(n_splits=number_splits, shuffle=True)

    y_test_arr = np.zeros((len(seeds), number_splits), dtype=object)
    y_pred_arr = np.zeros((len(seeds), number_splits), dtype=object)


    for cv, seed in enumerate(seeds):
        np.random.seed(seed)
        print(f"CV {cv}:")
        for run, (train_index, test_index) in enumerate(kf.split(x_data)):
            print(f"    Run {run}")
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False)
            crf.fit(x_train, y_train)

            y_pred = crf.predict(x_test)

            y_test_arr[cv][run] = list(y_test)
            y_pred_arr[cv][run] = list(y_pred)


    print("ALL: ")
    plot_results(y_test_arr, y_pred_arr)


def plot_results(y_test_arr, y_pred_arr):

    shape = y_pred_arr.shape
    num_seeds = shape[0]
    num_splits = shape[1]
    print(shape)
    for seed_id in range(num_seeds):
        all_y_test = []
        all_y_pred = []
        for split_id in range(num_splits):
            all_y_test = all_y_test + y_test_arr[seed_id, split_id]
            all_y_pred = all_y_pred + y_pred_arr[seed_id, split_id]
        print(f"CV {seed_id}:")
        print(metrics.flat_classification_report(all_y_test, all_y_pred, digits=3))
        print(metrics.flat_f1_score(all_y_test, all_y_pred, average='weighted'))

        
if __name__ == "__main__":
    main()
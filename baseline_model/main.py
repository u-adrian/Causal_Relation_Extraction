from baseline_model.data_loader import load_data

from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

from baseline_model.feature_creator import sent2features


def main():
    dataset_path = "C:/Dev/Smart_Data/E4/datasets/output.xlsx"

    x_data, y_data = load_data(dataset_path)
    x_data = [sent2features(s) for s in x_data]

    x_test = x_data[8000:]
    y_test = y_data[8000:]

    x_train = x_data[0:8000]
    y_train = y_data[0:8000]

    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=1000, all_possible_transitions=False)
    print("Start training ...")
    crf.fit(x_train, y_train)
    print("End of training")
    y_pred = crf.predict(x_test)

    print(metrics.flat_classification_report(y_test, y_pred, digits=3))

    print(metrics.flat_f1_score(y_test, y_pred, average='weighted'))


if __name__ == "__main__":
    main()
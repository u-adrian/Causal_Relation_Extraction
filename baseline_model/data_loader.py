import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics


def load_data():
    dataset_path = "C:/Dev/Smart_Data/E4/datasets/output.xlsx"

    with open(dataset_path, "rb") as data_file:
        df = pd.read_excel(data_file, index_col=0)

    print(tabulate(df.head(1), headers = 'keys', tablefmt = 'psql'))

    num_samples = len(df.index)
    x_train_and_test = [sent2better_format(df.iloc[sample]["context"], df.iloc[sample]["idx"], int(df.iloc[sample]["label"]), direction = int(df.iloc[sample]["direction"])) for sample in range(num_samples)]
    x_train_and_test = np.asarray(x_train_and_test, dtype=object)

    train_sentences = x_train_and_test[:, 0]
    y_train = list(x_train_and_test[:, 1])

    x_train = [sent2features(s) for s in train_sentences]

    x_test = x_train[8000:]
    y_test = y_train[8000:]

    x_train = x_train[0:8000]
    y_train = y_train[0:8000]


    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=False)

    crf.fit(x_train, y_train)
    y_pred = crf.predict(x_test)
    print(len(y_pred))
    print(metrics.flat_classification_report(
        y_test, y_pred, digits=3
    ))

    print(metrics.flat_f1_score(y_test, y_pred,
                          average='weighted'))
    

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        # 'word[-3:]': word[-3:],
        # 'word[-2:]': word[-2:],
        # 'word.isupper()': word.isupper(),
        # 'word.istitle()': word.istitle(),
        # 'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            # '-1:word.istitle()': word1.istitle(),
            # '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            # '+1:word.istitle()': word1.istitle(),
            # '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def sent2better_format(sent, idx, label, direction):
    context_list = sent[:-1].split(" ")

    # create start and end index for every word
    start_pos = []
    end_pos = []
    c = 0
    for con in context_list:
        start_pos.append(c)
        c += len(con)
        end_pos.append(c)
        c += 1
    start_end_pos_list = list(zip(start_pos, end_pos))

    span_1_tag = 'O'
    span_2_tag = 'O'
    if label == 1:
        if direction == 0:
            span_1_tag = 'C'
            span_2_tag = 'E'
        elif direction == 1:
            span_1_tag = 'E'
            span_2_tag = 'C'

    # create tag for every word
    span1_idx_list, span2_idx_list = extract_span_idx(idx)
    tags = []
    for start_end_pos in start_end_pos_list:
        if start_end_pos in span1_idx_list:
            tags.append(span_1_tag)
        elif start_end_pos in span2_idx_list:
            tags.append(span_2_tag)
        else:
            tags.append("O")

    return context_list, tags


def extract_span_idx(idx_string):
    idx_data = idx_string.split("\n")

    span1_idx_list = idx_data[0][6:].split(" ")
    span2_idx_list = idx_data[1][6:].split(" ")

    def splitting_func(idx):
        if idx == "":
            return
        start_end = idx.split(":")
        return int(start_end[0]), int(start_end[1])

    span1_idx_list = list(map(splitting_func, span1_idx_list))
    span2_idx_list = list(map(splitting_func, span2_idx_list))

    return span1_idx_list, span2_idx_list


def main():
    load_data()


if __name__ == "__main__":
    main()

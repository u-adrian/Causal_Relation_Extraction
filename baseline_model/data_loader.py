import numpy as np
import pandas as pd


def load_data(dataset_path):

    with open(dataset_path, "rb") as data_file:
        df = pd.read_excel(data_file, index_col=0)

    num_samples = len(df.index)
    print(num_samples)
    train_and_test_data = [extract_words_and_tags(df.iloc[sample]["context"], df.iloc[sample]["idx"],
                                                  int(df.iloc[sample]["label"]), int(df.iloc[sample]["direction"]))
                           for sample in range(num_samples)]

    train_and_test_data = np.asarray(train_and_test_data, dtype=object)

    x_data = list(train_and_test_data[:, 0])

    y_data = list(train_and_test_data[:, 1])

    return x_data, y_data


def extract_words_and_tags(sent, idx, label, direction):
    context_list = sent.split(" ")

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
        tag_appended = False
        for span1_idx in span1_idx_list:
            r = range(span1_idx[0], span1_idx[1])
            if start_end_pos[0] in r or start_end_pos[1] in r:
                tags.append(span_1_tag)
                tag_appended = True
                break
        if not tag_appended:
            for span2_idx in span2_idx_list:
                r = range(span2_idx[0], span2_idx[1])
                if start_end_pos[0] in r or start_end_pos[1] in r:
                    tags.append(span_2_tag)
                    tag_appended = True
                    break
        if not tag_appended:
            tags.append("O")
    assert len(context_list) == len(tags)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging
import re

tqdm.pandas(desc="prog_bar")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")


def process_text(text, mode="train"):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text) / 4)):
        sent = text[4 * i]
        relation = text[4 * i + 1]
        comment = text[4 * i + 2]
        blank = text[4 * i + 3]

        # check entries
        if mode == "train":
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
        assert re.match("^Comment", comment)
        assert len(blank) == 1

        sent = re.findall('"(.+)"', sent)[0]
        sent = re.sub("<e1>", "[E1]", sent)
        sent = re.sub("</e1>", "[/E1]", sent)
        sent = re.sub("<e2>", "[E2]", sent)
        sent = re.sub("</e2>", "[/E2]", sent)
        sents.append(sent)
        relations.append(relation), comments.append(comment)
        blanks.append(blank)
    return sents, relations, comments, blanks


def preprocess_semeval2010_8(args):
    """
    Data preprocessing for SemEval2010 task 8 dataset
    """
    data_path = (
        "./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    )
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, "r", encoding="utf8") as f:
        text = f.readlines()

    sents, relations, comments, blanks = process_text(text, "train")
    df_train = pd.DataFrame(data={"sents": sents, "relations": relations})

    data_path = "./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, "r", encoding="utf8") as f:
        text = f.readlines()

    sents, relations, comments, blanks = process_text(text, "test")
    df_test = pd.DataFrame(data={"sents": sents, "relations": relations})

    rm = Relations_Mapper(df_train["relations"])
    save_as_pickle("relations.pkl", rm)
    df_test["relations_id"] = df_test.progress_apply(
        lambda x: rm.rel2idx[x["relations"]], axis=1
    )
    df_train["relations_id"] = df_train.progress_apply(
        lambda x: rm.rel2idx[x["relations"]], axis=1
    )
    save_as_pickle("df_train.pkl", df_train)
    save_as_pickle("df_test.pkl", df_test)
    logger.info("Finished and saved!")

    return df_train, df_test, rm


def process_crest_text_alt(dataset_path):
    crest_df = pd.read_excel(
        dataset_path, usecols=["span1", "span2", "context", "label", "direction"]
    )
    crest_df["relations"] = pd.NA

    def augment_context(row):
        return (
            str(row["context"])
            .replace(row["span1"][2:-2], "[E1]" + row["span1"][2:-2] + "[/E1]")
            .replace(row["span2"][2:-2], "[E2]" + row["span2"][2:-2] + "[/E2]")
        )

    crest_df["sents"] = crest_df.apply(augment_context, axis=1)
    crest_df.drop(columns=["span1", "span2", "context"], inplace=True)

    def add_relation(row):
        if row["label"] == 0:
            return "Other"
        else:
            if row["direction"] == 0:
                return "Cause-Effect(e1, e2)"
            elif row["direction"] == 1:
                return "Cause-Effect(e2, e1)"
            else:
                return "Cause-Effect"

    crest_df["relations"] = crest_df.apply(add_relation, axis=1)
    crest_df.drop(columns=["label", "direction"], inplace=True)

    return crest_df


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


def process_crest_text(dataset_path):
    crest_df = pd.read_excel(
        dataset_path, usecols=["context", "idx", "label", "direction"]
    )
    crest_df["relations"] = pd.NA

    def augment_context(row):
        span1, span2 = extract_span_idx(row["idx"])
        return (
            row["context"][: span1[0][0]]
            + "[E1]"
            + row["context"][span1[0][0] : span1[0][1]]
            + "[/E1]"
            + row["context"][span1[0][1] : span2[0][0]]
            + "[E2]"
            + row["context"][span2[0][0] : span2[0][1]]
            + "[/E2]"
            + row["context"][span2[0][1] :]
        )

    crest_df["sents"] = crest_df.apply(augment_context, axis=1)
    crest_df.drop(columns=["context", "idx"], inplace=True)

    def add_relation(row):
        if row["label"] == 0:
            return "Other"
        else:
            if row["direction"] == 0:
                return "Cause-Effect(e1, e2)"
            elif row["direction"] == 1:
                return "Cause-Effect(e2, e1)"
            else:
                return "Cause-Effect"

    crest_df["relations"] = crest_df.apply(add_relation, axis=1)
    crest_df.drop(columns=["label", "direction"], inplace=True)

    return crest_df


def preprocess_crest_dataset(args):
    """Data preprocessing for CREST datasets"""
    dataset_path = "./data/CREST/crest.xlsx"
    logger.info("Reading training file %s..." % dataset_path)

    df = process_crest_text(dataset_path)

    rm = Relations_Mapper(df["relations"])
    save_as_pickle("relations.pkl", rm)
    df["relations_id"] = df.progress_apply(
        lambda row: rm.rel2idx[row["relations"]], axis=1
    )

    sents_train, sents_test, relations_train, relations_test = train_test_split(
        df[["sents", "relations"]], df["relations_id"], test_size=0.2, random_state=42
    )
    df_train = pd.concat([sents_train, relations_train], axis=1)
    df_test = pd.concat([sents_test, relations_test], axis=1)

    save_as_pickle("df_train.pkl", df_train)
    save_as_pickle("df_test.pkl", df_test)
    logger.info("Finished and saved!")

    return df_train, df_test, rm


class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}

        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1

        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key


class Pad_Sequence:
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """

    def __init__(
        self,
        seq_pad_value,
        label_pad_value=-1,
        label2_pad_value=-1,
    ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(
            seqs, batch_first=True, padding_value=self.seq_pad_value
        )
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_value
        )
        y_lengths = torch.LongTensor([len(x) for x in labels])

        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(
            labels2, batch_first=True, padding_value=self.label2_pad_value
        )
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        return (
            seqs_padded,
            labels_padded,
            labels2_padded,
            x_lengths,
            y_lengths,
            y2_lengths,
        )


def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = (
            [i for i, e in enumerate(x) if e == e1_id][0],
            [i for i, e in enumerate(x) if e == e2_id][0],
        )
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start


class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df["input"] = self.df.progress_apply(
            lambda x: tokenizer.encode(x["sents"]), axis=1
        )

        self.df["e1_e2_start"] = self.df.progress_apply(
            lambda x: get_e1e2_start(x["input"], e1_id=self.e1_id, e2_id=self.e2_id),
            axis=1,
        )
        print(
            "\nInvalid rows/total: %d/%d" % (df["e1_e2_start"].isnull().sum(), len(df))
        )
        self.df.dropna(axis=0, inplace=True)

    def __len__(
        self,
    ):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.df.iloc[idx]["input"]),
            torch.LongTensor(self.df.iloc[idx]["e1_e2_start"]),
            torch.LongTensor([self.df.iloc[idx]["relations_id"]]),
        )


def load_dataloaders(args):
    if args.model_no == 0:
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer

        model = args.model_size  #'bert-large-uncased' 'bert-base-uncased'
        lower_case = True
        model_name = "BERT"
    elif args.model_no == 1:
        from ..model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer

        model = args.model_size  #'albert-base-v2'
        lower_case = True
        model_name = "ALBERT"
    elif args.model_no == 2:
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer

        model = "bert-base-uncased"
        lower_case = False
        model_name = "BioBERT"

    if os.path.isfile("./data/%s_tokenizer.pkl" % model_name):
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info(
            "Pre-trained blanks tokenizer not found, initializing new tokenizer..."
        )
        if args.model_no == 2:
            tokenizer = Tokenizer(
                vocab_file="./additional_models/biobert_v1.1_pubmed/vocab.txt",
                do_lower_case=False,
            )
        else:
            tokenizer = Tokenizer.from_pretrained(model, do_lower_case=False)
        tokenizer.add_tokens(["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"])

        save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer)
        logger.info(
            "Saved %s tokenizer at ./data/%s_tokenizer.pkl" % (model_name, model_name)
        )

    relations_path = "./data/relations.pkl"
    train_path = "./data/df_train.pkl"
    test_path = "./data/df_test.pkl"
    if (
        os.path.isfile(relations_path)
        and os.path.isfile(train_path)
        and os.path.isfile(test_path)
    ):
        rm = load_pickle("relations.pkl")
        df_train = load_pickle("df_train.pkl")
        df_test = load_pickle("df_test.pkl")
        logger.info("Loaded preproccessed data.")
    else:
        if args.task == "semeval":
            df_train, df_test, rm = preprocess_semeval2010_8(args)
        elif args.task == "crest":
            df_train, df_test, rm = preprocess_crest_dataset(args)
        else:
            logger.info("Given task '%s' is unknown." % (args.task))

    return df_train, df_test, len(df_train), len(df_test)

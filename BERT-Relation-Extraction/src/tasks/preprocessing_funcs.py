#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
import sys

# TODO: Generalisieren
sys.path.append(
    "/Users/lukaskubelka/Documents/_KIT/_Studium/_M.Sc./_Semester/Semester-2/PSDA/Uebungen/E4/Causal_Relation_Extraction"
)
from baseline_model.data_loader import load_data
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")


def process_crest_text_alt(x_data, y_data):
    sents, relations = [], []
    for token_list, label_list in zip(x_data, y_data):
        if not any([label in ["C", "E"] for label in label_list]):
            relations.append("Other")
            # TODO: Hier etwas mit den Spans einfallen lassen
            sent = " ".join(token_list)
            sent = f"{sent}."
            sents.append(sent)
            continue

        first_span_found = False
        first_span_label = ""
        idx = 0
        for label in label_list:
            if label in ["C", "E"]:
                if idx == (len(label_list) - 1):
                    token_list[idx] = f"[E2]{token_list[idx]}[/E2]"
                    break
                if not first_span_found:
                    first_span_label = label
                    if label_list[idx + 1] != label:
                        token_list[idx] = f"[E1]{token_list[idx]}[/E1]"
                        first_span_found = True
                    else:
                        token_list[idx] = f"[E1]{token_list[idx]}"
                        while label_list[idx + 1] == label:
                            idx += 1
                        token_list[idx] = f"{token_list[idx]}[/E1]"
                else:
                    if label_list[idx + 1] != label:
                        token_list[idx] = f"[E2]{token_list[idx]}[/E2]"
                    else:
                        token_list[idx] = f"[E2]{token_list[idx]}"
                        while label_list[idx + 1] == label:
                            idx += 1
                        token_list[idx] = f"{token_list[idx]}[/E2]"
            idx += 1
        relations.append(
            "Cause-Effect(e1, e2)"
        ) if first_span_label == "C" else relations.append("Cause-Effect(e2, e1)")

        sent = " ".join(token_list)
        sent = f"{sent}."
        sents.append(sent)

    return sents, relations


def process_crest_text(dataset_path):
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


def preprocess_crest_dataset(args):
    """Data preprocessing for CREST datasets"""
    dataset_path = args.data  # "./data/CREST/crest.xlsx"
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

    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")
    assert e1_id != e2_id != 1

    # if args.task == "semeval":
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
        df_train, df_test, rm = preprocess_crest_dataset(args)
        # df_train, df_test, rm = preprocess_semeval2010_8(args)

    print(df_train)

    train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)

    print(train_set.df)
    print(test_set.df)

    train_length = len(train_set)
    test_length = len(test_set)
    PS = Pad_Sequence(
        seq_pad_value=tokenizer.pad_token_id,
        label_pad_value=tokenizer.pad_token_id,
        label2_pad_value=-1,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=PS,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=PS,
        pin_memory=False,
    )

    return train_loader, test_loader, train_length, test_length

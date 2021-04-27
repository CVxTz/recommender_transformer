import random

import numpy as np
import pandas as pd


def map_column(df: pd.DataFrame, col_name: str):
    """
    Maps column values to integers
    :param df:
    :param col_name:
    :return:
    """
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 1 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    df[col_name + "_mapped"] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def split_df(
    df: pd.DataFrame, split: str, history_size: int = 30, horizon_size: int = 5
):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows
    :param df:
    :param split:
    :param history_size:
    :param horizon_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(horizon_size + 1, df.shape[0] - horizon_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    label_index = end_index - horizon_size
    start_index = max(0, label_index - history_size)

    history = df[start_index:label_index]
    targets = df[label_index:end_index]

    return history, targets


def pad_arr(arr: np.ndarray, expected_size: int = 60):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def pad_list(list_integers, history_size: int, pad_val: int = 0):
    """

    :param list_integers:
    :param history_size:
    :param pad_val:
    :return:
    """

    if len(list_integers) < history_size:
        list_integers = [pad_val] * (history_size - len(list_integers)) + list_integers

    return list_integers

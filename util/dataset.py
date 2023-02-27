import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import OrderedDict


def read_plots_from_csv(filename, col_name_plots='plot', col_name_movie='name'):
    df = pd.read_csv(filename, sep='@')
    mask = df.notnull().all(axis=1)
    df_non_null_rows = df[mask]
    plots = df_non_null_rows[col_name_plots]
    movie_name = df_non_null_rows[col_name_movie]
    return OrderedDict(zip(list(movie_name.values), list(plots.values)))


def select_random_items(data, exclude_index, n=3):
    """
    Select n random items from a Pandas series object, excluding a given index.

    Args:
        data (pd.Series): the input Pandas series object
        exclude_index: the index of the item to exclude from the selection
        n (int): the number of items to select (default=3)

    Returns:
        pd.Series: a Pandas series object containing the selected items
    """
    # Create a list of all index values except the excluded index
    index_list = data.index.tolist()
    index_list.remove(exclude_index)

    # Select n random items from the index list and use them to index the original series
    random_indices = np.random.choice(index_list, size=n, replace=False)
    selected_items = data[random_indices]

    return list(selected_items.values)


class MoviePlotDataset(Dataset):
    def __init__(self, data_col_name, label_col_name):
        filename = os.path.join('data', 'movie_1000_clean.csv')
        self.df = pd.read_csv(filename, sep='@')
        self.data = self.df[data_col_name]
        self.label = self.df[label_col_name]
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos_plot = self.data[idx]
        neg_plot = select_random_items(self.data, idx, n=1)
        question = self.label[idx]
        pos_plot = self.tokenizer(pos_plot, max_length=512,
                                  padding='max_length', return_tensors='pt')
        neg_plot = self.tokenizer(neg_plot, max_length=512,
                                  padding='max_length', return_tensors='pt')
        question = self.tokenizer(
            question, max_length=512, padding='max_length', return_tensors='pt')
        p = pos_plot['input_ids'].squeeze(0)
        neg_p = neg_plot['input_ids'].squeeze(0)

        q = question['input_ids'].squeeze(0)
        return [p, neg_p], q


class MoviePlotDatasetSmall(Dataset):
    def __init__(self, data_col_name, label_col_name):
        filename = os.path.join('data', 'movie_1000_clean.csv')
        self.df = pd.read_csv(filename, sep='@')
        self.data = self.df[data_col_name][:1]
        self.label = self.df[label_col_name][:1]
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        plot = self.data[idx]
        question = self.label[idx]
        plot = self.tokenizer(plot, max_length=512,
                              padding='max_length', return_tensors='pt')
        question = self.tokenizer(
            question, max_length=512, padding='max_length', return_tensors='pt')
        p = plot['input_ids'].squeeze(0)
        q = question['input_ids'].squeeze(0)
        return p, q

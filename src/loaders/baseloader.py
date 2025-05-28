from torch.utils.data import Dataset
import numpy as np
import torch
from os import path as ospath
from pandas import read_csv as pdreader

class BaseDataset(Dataset):
    def __init__(self, data, dtype=torch.float32):        
        self.dtype = dtype

        self.data = data.type(dtype)

        self.missing = torch.isnan(data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        datarow = self.data[idx]
        missing_cols = self.missing[idx]
        datarow[missing_cols] = -1.
        fid_tensor = torch.tensor(idx)

        return dict(
            table = datarow.unsqueeze(0),
            miss_map = missing_cols,
            id = fid_tensor
        )

class ContextBaseDataset(Dataset):
    def __init__(self, data, embeddings, dtype=torch.float32):        
        self.dtype = dtype

        self.data = data.type(dtype)

        self.missing = torch.isnan(data)

        self.load_embeddings(embeddings)

    def load_embeddings(self, embeddings_file):
        loaded = np.load(embeddings_file)
        
        # Load all embeddings
        self.infoembeds = torch.tensor(loaded['colembed'], dtype=self.dtype)
        
        # Load all column names
        self.colnames = loaded['colnames']

        # Ensure embeddings match the number of columns in the data
        assert self.infoembeds.shape[0] == self.data.shape[1], "Number of embeddings does not match number of columns in data"

        print(f"Loaded {self.infoembeds.shape[0]} embeddings of dimension {self.infoembeds.shape[1]}")
        # print(f"First few column names: {self.colnames[:5]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        datarow = self.data[idx]
        missing_cols = self.missing[idx]
        datarow[missing_cols] = -1.
        fid_tensor = torch.tensor(idx)

        return dict(
            table = datarow.unsqueeze(0),
            miss_map = missing_cols,
            id = fid_tensor,
            info_embed = self.infoembeds
        )
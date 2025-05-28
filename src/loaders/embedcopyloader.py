from torch.utils.data import Dataset
import torch
import numpy as np

class EmbedCopyMaskedDataset(Dataset):
    def __init__(self, data, mask_ratio, embeddings, dtype=torch.float32):        
        self.dtype = dtype

        self.data = data.type(dtype)

        self.missing = torch.isnan(data)

        self.mask_ratio = mask_ratio

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

        ### Copy Masking
        # get an empty starting mask (no copy mask)
        mask_cols = torch.zeros(len(datarow), dtype=torch.bool)

        # grab one random from population
        if torch.rand(1).item() < self.mask_ratio:
            rnd_ind = torch.randint(self.data.shape[0], (1,)).item()
            mask_cols = self.missing[rnd_ind].clone()
            
        # safety check
        observed_cols = ~missing_cols
        if torch.dot(observed_cols.int(), (~mask_cols).int()) < 2:
            # fix when there's at most 1 observed feature, then flip back one
            mask_cols[(mask_cols & observed_cols)] = False

        # union of missing_cols and mask_cols
        mask_cols = missing_cols | mask_cols
        
        datarow[missing_cols] = -1.
        fid_tensor = torch.tensor(idx)

        return dict(
            table = datarow.unsqueeze(0),
            miss_map = missing_cols,
            mask_map = mask_cols,
            id = fid_tensor,
            info_embed = self.infoembeds
        )
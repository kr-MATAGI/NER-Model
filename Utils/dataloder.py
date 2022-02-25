import torch
from torch.utils.data import Dataset
import numpy as np

class NE_Datasets(Dataset):
    def __init__(self, path: str=""):
        self.attention_mask = np.load(path+"/attention_mask.npy")
        self.input_ids = np.load(path+"/input_ids.npy")
        self.labels = np.load(path+"/labels.npy")
        self.token_type_ids = np.load(path+"/token_type_ids.npy")

        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)

    def __len__(self):
        return len(self.attention_mask)

    def __getitem__(self, idx):
        items = {
            "attention_mask": self.attention_mask[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "token_type_ids": self.token_type_ids[idx]
        }

        return items

### TEST ###
if "__main__" == __name__:
    print("[Model.dataloader.py] -----TEST")
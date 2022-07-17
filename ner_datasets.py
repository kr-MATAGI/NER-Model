import numpy as np
import torch
from torch.utils.data import Dataset


### BERT





### ELECTRA
# electra-pos-lstm
class NER_POS_Dataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: np.ndarray, pos_data: np.ndarray):
        self.input_ids = data[:][:, :, 0]
        self.labels = data[:][:, :, 1]
        self.attention_mask = data[:][:, :, 2]
        self.token_type_ids = data[:][:, :, 3]
        self.pos_tag = pos_data
        self.input_seq_len = seq_len

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.input_seq_len = torch.tensor(self.input_seq_len, dtype=torch.long)
        self.pos_tag = torch.tensor(self.pos_tag, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "attention_mask": self.attention_mask[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "token_type_ids": self.token_type_ids[idx],
            "input_seq_len": self.input_seq_len[idx],
            "pos_tag_ids": self.pos_tag[idx]
        }

        return items
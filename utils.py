import numpy as np
import pandas as pd
import os
import torch
import time, sys
from torch.utils.data import DataLoader

class chaiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def preprocess_data(tokenizer,data):
    encodings = tokenizer(list(data["context"]), list(data["question"]), truncation=True, padding=True)
    start_positions = []
    end_positions = []
    id_ = []

    for i in range(len(data["answer_start"])):
        start_positions.append(encodings.char_to_token(i,data["answer_start"][i]))
        end_positions.append(encodings.char_to_token( i,(data["answer_start"][i] + len(data['answer_text'][i]) - 1) ))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length


    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    return encodings

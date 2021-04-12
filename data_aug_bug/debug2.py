import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __getitem__(self, index):
        return torch.randint(0,1000,(3,)) #fix #2. sample random numbers using PyTorch. Dont use numpy at all

    def __len__(self):
        return 16 

dataset = RandomDataset()
dataloader = DataLoader(dataset, batch_size=2, num_workers=4)  
for batch in dataloader:
    print(batch)

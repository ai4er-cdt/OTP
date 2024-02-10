from typing import Tuple
import torch as t
from torch.utils.data import Dataset


class SimDataset(Dataset):
    """
    TODO: 
    - fold in all data loading, preprocessing and reshaping.
    - train/val/test split.
    """
    def __init__(self,
                 X: t.Tensor,
                 Y: t.Tensor,
                 device: str):
        super().__init__()
        self.X = X; self.Y = Y
        self.device = device
    
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, ix: int) -> Tuple[t.Tensor, t.Tensor]:
        return self.X[ix].to(self.device), self.Y[ix].to(self.device)  
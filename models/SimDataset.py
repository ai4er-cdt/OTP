from typing import Callable, Any, List, Literal, Tuple, Optional
import torch as t
from torch.utils.data import Dataset, DataLoader


device = "cuda" if t.cuda.is_available() else "cpu"

class SimDataset(Dataset):
    """
    TODO: fold in all data loading, preprocessing and reshaping.
    """
    def __init__(self,
                 X: t.Tensor,
                 Y: t.Tensor):
        super().__init__()
        self.X = X; self.Y = Y
    
    def __len__(self) -> int:
        return len(X)

    def __getitem__(self, ix: int) -> Tuple[t.Tensor, t.Tensor]:
        x = self.X[ix].to(device)
        y = self.Y[ix].to(device)
        return x, y      
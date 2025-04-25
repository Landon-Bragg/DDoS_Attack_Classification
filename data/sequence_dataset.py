import os
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Loads .pt tensors of shape [C, D, H, W] and labels from directory structure:
    root/train/Normal, root/train/DDoS, root/val/Normal, root/val/DDoS
    """
    def __init__(self, split_dir, transform=None):
        self.samples = []
        classes = ['Normal', 'DDoS']
        for label, cls in enumerate(classes):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.endswith('.pt'):
                    path = os.path.join(cls_path, fname)
                    self.samples.append((path, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = torch.load(path)  # tensor [C, D, H, W]
        if self.transform:
            seq = self.transform(seq)
        return seq, label
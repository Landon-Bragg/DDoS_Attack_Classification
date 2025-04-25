import os, torch
from torch.utils.data import Dataset

class PreprocessedSequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples=[]
        self.classes=["Normal","DDoS"]
        for idx,cls in enumerate(self.classes):
            p=os.path.join(root_dir,cls)
            if not os.path.isdir(p): continue
            for f in os.listdir(p):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(p,f),idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        path,label=self.samples[idx]
        return torch.load(path), torch.tensor(label)

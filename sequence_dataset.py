import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class TrafficSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_length=8):
        self.root = root_dir
        self.transform = transform
        self.seq_length = seq_length
        self.classes = ['normal', 'ddos']
        self.samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            seq_folders = sorted([f for f in os.listdir(class_path) 
                                if f.startswith('seq_')])
            
            for seq_folder in seq_folders:
                seq_path = os.path.join(class_path, seq_folder)
                frame_paths = sorted(
                    [os.path.join(seq_path, f) 
                    for f in os.listdir(seq_path) 
                    if f.endswith('.png')],
                    key=lambda x: int(x.split('_')[-1].split('.')[0])
                ) 

                if len(frame_paths) == seq_length:
                    self.samples.append((frame_paths, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        sequence = []
        
        for path in frame_paths:
            with Image.open(path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                sequence.append(img)
        
        # Stack as (B, C, D, H, W) format where D is temporal dimension (sequence length)
        # Your model expects input as [batch, channels, depth, height, width]
        sequence_tensor = torch.stack(sequence, dim=0)  # [seq_len, channels, height, width]
        sequence_tensor = sequence_tensor.permute(1, 0, 2, 3)  # [channels, seq_len, height, width]
        
        return sequence_tensor, torch.tensor(label)
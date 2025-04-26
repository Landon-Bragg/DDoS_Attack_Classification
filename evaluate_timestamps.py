import os
import torch
from torch.utils.data import DataLoader, Dataset
from models.cnn3d import DDoS3DCNN
from data.sequence_dataset import SequenceDataset

# Custom wrapper to select only one frame at a time
def load_model(model_path, device):
    model = DDoS3DCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

class SingleTimeStepDataset(Dataset):
    def __init__(self, base_dataset, timestep):
        self.base_dataset = base_dataset
        self.timestep = timestep

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        seq, label = self.base_dataset[idx]  # seq: [C, D, H, W]
        frame = seq[:, self.timestep, :, :]   # frame: [C, H, W]
        frame = frame.unsqueeze(1)            # [C, 1, H, W]
        frame = frame.repeat(1, 8, 1, 1)       # [C, 8, H, W] --> repeat along D dimension
        return frame, label


def evaluate_model_at_timesteps(model_path, data_root, batch_size=16, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    base_dataset = SequenceDataset(os.path.join(data_root, 'val'))

    results = []

    for t in range(8):
        print(f"Evaluating at time t={t}")
        t_dataset = SingleTimeStepDataset(base_dataset, timestep=t)
        t_loader = DataLoader(t_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=(device.type == 'cuda'))

        all_preds, all_labels = [], []

        for seqs, labels in t_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(seqs)
                preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        total = len(all_labels)
        acc = correct / total
        print(f"Accuracy at t={t}: {acc:.4f}\n")
        results.append((t, acc))

    return results

if __name__ == '__main__':
    model_path = 'outputs/model_clean.pth'
    data_root = 'preprocessed_dataset'

    evaluate_model_at_timesteps(model_path, data_root)

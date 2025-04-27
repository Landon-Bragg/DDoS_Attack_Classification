import os
import torch
from torch.utils.data import DataLoader, Dataset
from models.cnn3d import DDoS3DCNN
from data.sequence_dataset import SequenceDataset
from attacks.adversarial_attacks import get_fgsm, get_pgd
import torchvision.transforms.functional as F
import random

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
        frame = seq[:, self.timestep, :, :]   # [C, H, W]
        frame = frame.unsqueeze(1)            # [C, 1, H, W]
        frame = frame.repeat(1, 8, 1, 1)       # [C, 8, H, W]
        return frame, label

def augment_batch(seqs):
    B, C, D, H, W = seqs.shape
    out = []
    for b in range(B):
        frames = []
        for i in range(D):
            frame = seqs[b, :, i]  # [C, H, W]
            angle = random.uniform(-18, 18)
            frame = F.rotate(frame, angle)
            shear = random.uniform(-11, 11)
            frame = F.affine(frame, angle=0, translate=[0,0], scale=1.0, shear=shear)
            crop_factor = random.uniform(0.75, 1.0)
            new_h, new_w = int(H * crop_factor), int(W * crop_factor)
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)
            frame = F.resized_crop(frame, top, left, new_h, new_w, [H, W])
            noise = torch.randn_like(frame) * 0.17
            frame = torch.clamp(frame + noise, 0, 1)
            frames.append(frame)
        out.append(torch.stack(frames, dim=1))  # [C, D, H, W]
    return torch.stack(out, dim=0) 

def evaluate_model_adv(model_path, data_root, batch_size=16, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    atk_pgd = get_pgd(model, eps=1.225, alpha=1.1, steps=40)
    atk_fgsm = get_fgsm(model, eps=1.19)

    base_dataset = SequenceDataset(os.path.join(data_root, 'val'))

    modes = ['aug', 'pgd', 'fgsm']
    results = []

    for mode in modes:
        print(f"\n==== Evaluating {mode.upper()} ====")

        for t in range(8):
            print(f"Evaluating {mode.upper()} at time t={t}")

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

                if mode == 'aug':
                    seqs = augment_batch(seqs)
                elif mode == 'pgd':
                    seqs = atk_pgd(seqs.requires_grad_(True), labels)
                elif mode == 'fgsm':
                    seqs = atk_fgsm(seqs.requires_grad_(True), labels)

                with torch.no_grad():
                    outputs = model(seqs)
                    preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            correct = sum(p == l for p, l in zip(all_preds, all_labels))
            total = len(all_labels)
            acc = correct / total
            print(f"Accuracy at t={t}: {acc:.4f}\n")
            results.append((mode, t, acc))

    return results

if __name__ == '__main__':
    model_path = 'outputs/model_adversarial.pth'
    data_root = 'preprocessed_dataset'

    evaluate_model_adv(model_path, data_root)

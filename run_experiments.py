import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import pandas as pd
from data.sequence_dataset import SequenceDataset
from models.cnn3d import DDoS3DCNN
from attacks.adversarial_attacks import get_fgsm, get_pgd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix

from utils import set_seed
import config
import torchvision.transforms.functional as F
import random
from torch.amp import autocast, GradScaler

import torch.backends.cudnn as cudnn
# cuDNN autotuner (idrk what this does, supposed to make it go faster)
cudnn.benchmark = True

def augment_batch(seqs):
    """
    seqs: Tensor of shape [B, C, D, H, W]
    returns: same shape, each frame randomly rotated, sheared, zoomed, and noised
    """
    B, C, D, H, W = seqs.shape
    out = []
    for b in range(B):
        frames = []
        for i in range(D):
            frame = seqs[b, :, i]  # [C, H, W]
            # rotation
            angle = random.uniform(-23, 23)
            frame = F.rotate(frame, angle)
            # shear
            shear = random.uniform(-10, 10)
            frame = F.affine(frame, angle=0, translate=[0,0], scale=1.0, shear=shear)
            # zoom via random crop + resize
            crop_factor = random.uniform(0.72, 1.0)
            new_h, new_w = int(H * crop_factor), int(W * crop_factor)
            top  = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)
            frame = F.resized_crop(frame, top, left, new_h, new_w, [H, W])
            # noise
            noise = torch.randn_like(frame) * 0.18
            frame = torch.clamp(frame + noise, 0, 1)
            frames.append(frame)
        out.append(torch.stack(frames, dim=1))  # [C, D, H, W]
    return torch.stack(out, dim=0)  # [B, C, D, H, W]

def train_one_epoch(model, loader, optimizer, criterion, device, regime, cfg, scaler):
    print(f"--- Training regime: {regime} ---")
    model.train()
    running_loss = 0

    for batch_idx, (seqs, labels) in enumerate(loader, start=1):
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()

        if regime == 'adversarial':
            b = seqs.size(0)
            # compute slice sizes
            n_clean = max(1, int(b * 0.1))
            n_aug   = max(1, int(b * 0.12))
            n_pgd   = max(1, int(b * 0.28))
            # remaining for FGSM
            n_fgsm  = b - (n_clean + n_aug + n_pgd)

            # slice off each chunk
            c_seqs  = seqs[:n_clean];      c_lbl = labels[:n_clean]
            a_seqs  = seqs[n_clean:n_clean+n_aug]
            a_lbl   = labels[n_clean:n_clean+n_aug]
            p_seqs  = seqs[n_clean+n_aug : n_clean+n_aug+n_pgd]
            p_lbl   = labels[n_clean+n_aug : n_clean+n_aug+n_pgd]
            f_seqs  = seqs[-n_fgsm:];      f_lbl = labels[-n_fgsm:]

            # apply attacks
            aug_seqs = augment_batch(a_seqs)
            pgd_seqs = get_pgd(model, cfg['pgd_eps'], cfg['pgd_alpha'], cfg['pgd_steps'])(p_seqs, p_lbl)
            fgs_seqs = get_fgsm(model, cfg['fgsm_eps'])(f_seqs, f_lbl)

            # concatenate back
            seqs_batch = torch.cat([c_seqs, aug_seqs, pgd_seqs, fgs_seqs], dim=0)
            labels_batch = torch.cat([c_lbl, a_lbl, p_lbl, f_lbl], dim=0)

        else:
            seqs_batch, labels_batch = seqs, labels

        # mixed‑precision forward/backward
        with autocast(device_type=device.type):
            outputs = model(seqs_batch)
            loss    = criterion(outputs, labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} — Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    print(f"Epoch complete — Avg Loss: {avg_loss:.4f}\n")
    return avg_loss

def evaluate(model, loader, device, cfg, mode):
    """
    Evaluate under clean, aug, pgd, or fgsm.
    Adversarial examples are generated before the no_grad inference.
    Returns accuracy, precision, recall, and AUC.
    """
    print(f"--- Evaluating condition: {mode} ---")
    model.eval()
    atk_pgd  = get_pgd(model, cfg['pgd_eps'], cfg['pgd_alpha'], cfg['pgd_steps'])
    atk_fgsm = get_fgsm(model, cfg['fgsm_eps'])
    all_preds, all_probs, all_labels = [], [], []

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)

        # 1) Generate the evaluation batch
        if mode == 'clean':
            seqs_eval = seqs

        elif mode == 'aug':
            seqs_eval = augment_batch(seqs)

        elif mode == 'pgd':
            seqs_grad = seqs.clone().detach().requires_grad_(True)
            seqs_eval = atk_pgd(seqs_grad, labels)

        elif mode == 'fgsm':
            seqs_grad = seqs.clone().detach().requires_grad_(True)
            seqs_eval = atk_fgsm(seqs_grad, labels)

        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

        # 2) Inference under no_grad
        with torch.no_grad():
            outputs = model(seqs_eval)
            # Get class probabilities using softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

        # Store predictions, probabilities for positive class (for AUC), and labels
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # Assuming class 1 is positive
        all_labels.extend(labels.cpu().tolist())

    # Needs to clear cache to prevent gpu overuse, use outside loop to speed things up
    torch.cuda.empty_cache()   

    # Calculate metrics
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    # Calculate AUC (need to handle cases where only one class is present)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # This happens when only one class is present in the labels
        auc = float('nan')
    
    # Print confusion matrix for additional insight
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix [{mode}]:\n{cm}")
    
    print(f"Accuracy [{mode}]: {acc:.4f}")
    print(f"Precision [{mode}]: {precision:.4f}")
    print(f"Recall [{mode}]: {recall:.4f}")
    print(f"AUC [{mode}]: {auc:.4f}\n")
    
    return acc, precision, recall, auc


def main():
    cfg    = config.CONFIG
    set_seed(cfg['seed'])

    # prepare device, DataLoaders, etc.
    try:
        device = torch.device('cuda'); _ = torch.zeros(1, device=device)
    except:
        device = torch.device('cpu')
    use_cuda = (device.type == 'cuda')

    train_ds = SequenceDataset(os.path.join(cfg['data_root'], 'train'))
    val_ds   = SequenceDataset(os.path.join(cfg['data_root'], 'val'))

    train_loader = DataLoader(train_ds,
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              num_workers=cfg['num_workers'],
                              pin_memory=use_cuda,
                              persistent_workers=use_cuda)
    val_loader = DataLoader(val_ds,
                            batch_size=cfg['batch_size'],
                            shuffle=False,
                            num_workers=cfg['num_workers'],
                            pin_memory=use_cuda,
                            persistent_workers=use_cuda)

    results = []

    for regime in ['clean', 'adversarial']:
        print(f"\n===== Starting regime: {regime} =====")
        model     = DDoS3DCNN(cfg['num_classes']).to(device)
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg['lr'],
                                weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min',
                        factor=0.5, patience=3)
        criterion = nn.CrossEntropyLoss()
        scaler    = GradScaler(device=device.type)

        if regime == 'clean':
            # run *all* epochs
            for epoch in range(cfg['epochs']):
                print(f"[{regime}] Epoch {epoch+1}/{cfg['epochs']}")
                _ = train_one_epoch(model, train_loader,
                                    optimizer, criterion,
                                    device, regime, cfg, scaler)

        else:
            # adversarial: early-stop when LR has been cut N times
            prev_lr = cfg['lr']
            lr_cuts = 0
            max_cuts = 4
            for epoch in range(cfg['epochs']):
                print(f"[{regime}] Epoch {epoch+1}/{cfg['epochs']}")
                train_loss = train_one_epoch(model, train_loader,
                                             optimizer, criterion,
                                             device, regime, cfg, scaler)
                scheduler.step(train_loss)

                curr_lr = optimizer.param_groups[0]['lr']
                print(f"  → LR now: {curr_lr:.2e}")
                if curr_lr < prev_lr:
                    lr_cuts += 1
                    prev_lr = curr_lr
                    print(f"  ⚡️ LR cut #{lr_cuts}")
                if lr_cuts >= max_cuts:
                    print(f"⏹ Adversarial LR converged after {lr_cuts} cuts, stopping early.")
                    break

        # save once per regime
        torch.save(model.state_dict(),
                   os.path.join(cfg['output_dir'], f"model_{regime}.pth"))

        # single evaluation pass *after* training
        for mode in ['clean', 'aug', 'pgd', 'fgsm']:
            acc, precision, recall, auc = evaluate(model, val_loader, device, cfg, mode)
            results.append({
                'Regime':    regime,
                'Condition': mode,
                'Accuracy':  acc,
                'Precision': precision,
                'Recall':    recall,
                'AUC':       auc
            })

    # finally, dump your DataFrame
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg['output_dir'], 'results.csv'),
              index=False)
    
    # Print formatted results
    print("All experiments complete:")
    print(df.to_string(float_format="{:.4f}".format))
    
    # Optionally create a more visual results table
    from matplotlib import pyplot as plt
    import seaborn as sns
            
    # Create a heatmap to visualize the results
    plt.figure(figsize=(14, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        pivot_df = df.pivot(index='Regime', columns='Condition', values=metric)
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f', cbar=True)
        plt.title(f'{metric} by Training Regime and Test Condition')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['output_dir'], 'metrics_visualization.png'))
    print(f"Visualization saved to {os.path.join(cfg['output_dir'], 'metrics_visualization.png')}")


if __name__ == '__main__':
    main()

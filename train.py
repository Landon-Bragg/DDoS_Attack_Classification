import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score

from torch.utils.tensorboard import SummaryWriter
from torchattacks import PGD, FGSM

from sequence_dataset import TrafficSequenceDataset

class DDoS3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DDoS3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        self.linear_input_size = 256 * 2 * 50 * 50
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Always use clean transforms for training.
        self.train_set = TrafficSequenceDataset(
            os.path.join(config["data_root"], "train"),
            transform=self._get_clean_transforms(),
            seq_length=config.get("sequence_length", 8)
        )
        self.val_set = TrafficSequenceDataset(
            os.path.join(config["data_root"], "val"),
            transform=self._get_clean_transforms(),
            seq_length=config.get("sequence_length", 8)
        )
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"]
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"]
        )

        # Model setup
        self.model = DDoS3DCNN(num_classes=2)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # Training components
        self.optimizer = Adam(self.model.parameters(), lr=config["lr"])
        self.criterion = nn.CrossEntropyLoss()

        # For adversarial evaluation, default attacker is PGD.
        self.attacker = PGD(
            self.model,
            eps=config["pgd_epsilon"],
            alpha=config["pgd_alpha"],
            steps=config["pgd_steps"],
            random_start=config["pgd_random_start"]
        )
        
        # GradScaler for mixed precision training
        self.scaler = torch.amp.GradScaler()
        # TensorBoard
        self.writer = SummaryWriter(log_dir="runs/ddos_l40_v1")
        self.current_epoch = 0

    def _get_clean_transforms(self):
        # Clean training/evaluation transforms (resize, to tensor, normalize)
        transform_list = [
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform_list)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for sequences, labels in tqdm(self.train_loader, desc="Training"):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            # If augmented training mode is selected, mix in clean, augmented, and adversarial examples.
            if self.config.get("training_mode", "clean") == "augmented":
                batch_size = sequences.shape[0]
                rand = torch.rand(batch_size, device=self.device)
                p_clean = 0.15
                p_aug = 0.3  # then p_adv is 1 - p_clean - p_aug
                clean_indices = (rand < p_clean).nonzero(as_tuple=True)[0]
                aug_indices = ((rand >= p_clean) & (rand < p_clean + p_aug)).nonzero(as_tuple=True)[0]
                adv_indices = (rand >= p_clean + p_aug).nonzero(as_tuple=True)[0]

                new_sequences = []
                new_labels = []
                if clean_indices.numel() > 0:
                    new_sequences.append(sequences[clean_indices])
                    new_labels.append(labels[clean_indices])
                if aug_indices.numel() > 0:
                    aug_batch = self.apply_augmentation(sequences[aug_indices])
                    new_sequences.append(aug_batch)
                    new_labels.append(labels[aug_indices])
                if adv_indices.numel() > 0:
                    sequences_adv = sequences[adv_indices]
                    sequences_adv.requires_grad_()
                    adv_batch = self.attacker(sequences_adv, labels[adv_indices])
                    new_sequences.append(adv_batch)
                    new_labels.append(labels[adv_indices])
                if len(new_sequences) > 0:
                    sequences = torch.cat(new_sequences, dim=0)
                    labels = torch.cat(new_labels, dim=0)
                else:
                    print("Warning: Augmented batch empty, using clean batch")

            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total_samples * 100
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train', accuracy, epoch)
        return avg_loss, accuracy





    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for sequences, labels in tqdm(self.val_loader, desc="Validation"):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total_samples * 100
        self.writer.add_scalar('Loss/val', avg_loss, self.current_epoch)
        self.writer.add_scalar('Accuracy/val', accuracy, self.current_epoch)
        return avg_loss, accuracy

    def evaluate_model(self, evaluation_mode="clean"):
        """
        evaluation_mode:
          - "clean": evaluate on clean data.
          - "adversarial": evaluate on adversarially perturbed data (default attacker).
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        for sequences, labels in self.val_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            if evaluation_mode == "adversarial":
                # Ensure gradients are tracked for adversarial attack.
                sequences.requires_grad_()
                sequences = self.attacker(sequences, labels)
            with torch.no_grad():
                outputs = self.model(sequences)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds) * 100
        prec = precision_score(all_labels, all_preds, average='weighted')
        rec = recall_score(all_labels, all_preds, average='weighted')
        return acc, prec, rec

    def evaluate_augmented(self):
        """
        Evaluate the model on augmented data. For each sequence in the validation set,
        we apply a series of non-gradient-based augmentations (skew, rotate, noise, zoom)
        before forwarding it through the model.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        for sequences, labels in self.val_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            sequences_aug = self.apply_augmentation(sequences)
            with torch.no_grad():
                outputs = self.model(sequences_aug)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds) * 100
        prec = precision_score(all_labels, all_preds, average='weighted')
        rec = recall_score(all_labels, all_preds, average='weighted')
        return acc, prec, rec

    def apply_augmentation(self, sequences):
        """
        Applies a series of augmentations (rotation, skew, noise, zoom) to each frame in the sequence.
        sequences: tensor of shape [batch, channels, depth, height, width]
        """
        import torchvision.transforms.functional as F
        import random
        import torch
        batch_aug = []
        for seq in sequences:
            # seq: [channels, depth, height, width]
            c, d, h, w = seq.shape
            frames = []
            for i in range(d):
                frame = seq[:, i, :, :]  # shape: [c, h, w]
                # Apply random rotation with a minimum magnitude.
                angle_magnitude = random.uniform(30, 60)
                angle = angle_magnitude if random.random() < 0.5 else -angle_magnitude
                frame = F.rotate(frame, angle)
                # Apply random skew (using affine with shear) with a minimum magnitude.
                shear_magnitude = random.uniform(15, 30)
                shear = shear_magnitude if random.random() < 0.5 else -shear_magnitude
                frame = F.affine(frame, angle=0, translate=[0, 0], scale=1.0, shear=shear)
                # Simulate zoom via random crop then resize back.
                crop_factor = random.uniform(0.6, 1.0)
                new_h = int(h * crop_factor)
                new_w = int(w * crop_factor)
                top = random.randint(0, h - new_h) if h - new_h > 0 else 0
                left = random.randint(0, w - new_w) if w - new_w > 0 else 0
                frame = F.resized_crop(frame, top=top, left=left, height=new_h, width=new_w, size=[h, w])
                # Add random Gaussian noise.
                noise = torch.randn_like(frame) * 0.35
                frame = frame + noise
                # Clamp pixel values to ensure they remain in the valid range.
                frame = torch.clamp(frame, 0, 1)
                frames.append(frame)
            seq_aug = torch.stack(frames, dim=1)  # shape: [c, depth, h, w]
            batch_aug.append(seq_aug)
        return torch.stack(batch_aug)


    def save_checkpoint(self, epoch, is_best=False):
        state_dict_to_save = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        state = {
            "epoch": epoch,
            "state_dict": state_dict_to_save,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config
        }
        checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")
        try:
            torch.save(state, filename, _use_new_zipfile_serialization=False)
            print(f"Checkpoint saved to {filename}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def run(self):
        best_val_acc = -1
        start_epoch = 0
        for epoch in range(start_epoch, self.config["epochs"]):
            self.current_epoch = epoch
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch, is_best=False)
        self.writer.close()
        print("Training finished.")

# Default configuration
config = {
    "data_root": r"C:\Users\cybears\Downloads\DDoS\dataset_root",  
    "batch_size": 16,
    "lr": 3e-4,
    "epochs": 3,
    "pgd_epsilon": 0.75,
    "pgd_alpha": 0.1,
    "pgd_steps": 100,
    "pgd_random_start": True,
    "num_workers": 4, 
    "pin_memory": True,
    "sequence_length": 8,
    "checkpoint_dir": "checkpoints",
    "training_mode": "clean"  # or "augmented
    }

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
    trainer = Trainer(config)
    trainer.run()

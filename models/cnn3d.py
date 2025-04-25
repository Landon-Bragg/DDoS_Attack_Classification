import torch.nn as nn

# This is our model
class DDoS3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
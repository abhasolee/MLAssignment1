import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Architecture 1: MLP ---
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.3),    
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# --- Architecture 2: CNN  ---
class CNN(nn.Module):
    def __init__(self, input_channels, image_size, num_classes, is_1d=False):
        super(CNN, self).__init__()
        self.is_1d = is_1d
        
        if is_1d:
            self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2) 
            self.fc_input = 64 * (image_size // 4) 
        else:
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc_input = 64 * (image_size // 4) * (image_size // 4)

        self.fc = nn.Linear(self.fc_input, num_classes)

    def forward(self, x):
        if self.is_1d:
            if x.dim() == 2:
                x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Architecture 3: Attention/Transformer ---
class SimpleAttention(nn.Module):
    """
    A simplified attention mechanism suitable for both tabular (as token sequence) 
    or flattened images.
    """
    def __init__(self, input_dim, num_classes):
        super(SimpleAttention, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.embedding(x) 
        x = x.unsqueeze(1) 
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        return self.fc(x)
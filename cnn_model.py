import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 
from torch.utils.data import Dataset
from torchvision import transforms
import os

class FingerprintTextureDataset(Dataset):
    def __init__(self, pairs, core_finder_func, augment=False):
        self.pairs = pairs
        self.core_finder = core_finder_func 
        # Resolution must be 96 for the STN calculations below
        self.img_size = 96 # hard coded
        self.augment = augment

        if self.augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=30),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1),
                    shear=10
                ),
                transforms.ToTensor()
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs)
        
    def preprocess_image(self,img_path):

        if img_path is None or not os.path.exists(img_path):
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)

        img_raw = cv2.imread(img_path, 0)
        if img_raw is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)
        img_float = img_raw.astype(np.float32)
        mean = np.mean(img_float)
        std = np.std(img_float) + 1e-6 # avoid div by zero
        img_norm = (img_float - mean) / std


        

        # Center of Mass Alignment
        try:
            
            M = cv2.moments(img_raw)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cy, cx = img_raw.shape[0]//2, img_raw.shape[1]//2

        except:
            cy, cx = img_raw.shape[0]//2, img_raw.shape[1]//2
            
        # Padding & Cropping
        half = self.img_size // 2
        padded = np.pad(img_norm, ((half, half), (half, half)), mode='constant', constant_values=0)
        cy += half
        cx += half
        patch = padded[cy-half:cy+half, cx-half:cx+half]
        

        patch = np.clip(patch, -3.0, 3.0)
        return patch

    def __getitem__(self, idx):
        g1, g2, label = self.pairs[idx]
        p1 = self.preprocess_image(g1.img_path)
        p2 = self.preprocess_image(g2.img_path)
        
        if self.augment:
            p1_min, p1_max = p1.min(), p1.max()
            p2_min, p2_max = p2.min(), p2.max()
            
            p1_u8 = ((p1 - p1_min) / (p1_max - p1_min + 1e-6) * 255).astype(np.uint8)
            p2_u8 = ((p2 - p2_min) / (p2_max - p2_min + 1e-6) * 255).astype(np.uint8)

            img1 = self.transform(p1_u8)
            img2 = self.transform(p2_u8)

        else:
             img1 = torch.from_numpy(p1).unsqueeze(0)
             img2 = torch.from_numpy(p2).unsqueeze(0)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# --- NEW STN MODULE ---
class STN_Module(nn.Module):
    def __init__(self):
        super(STN_Module, self).__init__()
        # Localization net
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),    # 96 -> 90
            nn.MaxPool2d(2, stride=2),         # 90 -> 45
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),   # 45 -> 41
            nn.MaxPool2d(2, stride=2),         # 41 -> 20
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        # Input size calculation: 10 channels * 20 height * 20 width = 4000
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 20 * 20, 32), # CORRECTED SIZE FOR 96x96 IMAGES
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 20 * 20) # Flatten
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        
        # --- INTEGRATE STN HERE ---
        self.stn = STN_Module()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 64, stride=2) 
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1)) 
        
        self.embedder = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.LeakyReLU(0.1), 
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def _make_layer(self, in_dim, out_dim, stride):
        return nn.Sequential(
            ResidualBlock(in_dim, out_dim, stride),
            ResidualBlock(out_dim, out_dim, stride=1)
        )

    def forward_one(self, x):
        # --- APPLY STN BEFORE CNN ---
        x = self.stn(x)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        
        diff = torch.abs(emb1 - emb2)
        combined = torch.cat((emb1, emb2, diff), dim=1)
        
        logits = self.classifier(combined)
        return logits.squeeze()

class EarlyStopping:
    def __init__(self, patience=12, delta=0.001, path='best_cnn.pth'):
        self.patience = patience 
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'   EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
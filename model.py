# film_layers_experiments.py


import os
import re
import random
import math
from collections import defaultdict
import time
import copy
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

# Attempt to import timm, otherwise use torchvision.models efficientnet_v2
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False
    from torchvision import models as tv_models

# ------------------------ Config ------------------------
ROOT_DIR = r"D:\project\A_model\data"  # <-- 修改为你的数据集根目录
SAVE_DIR = "./evaluation_results"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BACKBONE_NAME = "tf_efficientnetv2_s"  # timm name; if timm not available, fallback to torchvision efficientnet_v2_s
USE_TIMM = TIMM_AVAILABLE

NUM_EPOCHS = 12  # 可按需调整
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Experiment configs
FILM_LAYER_LIST = [1, 2, 3, 4, 5]  # vector FiLM layer counts to try
RUN_FEATURE_MAP_FILM = True         # also run feature-map-wise FiLM as a separate experiment

# For Grad-CAM visualization: number of val samples to save cams for each run
NUM_GRADCAM_SAMPLES = 4

# ------------------------ Reproducibility ------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------ Dataset ------------------------
class CoalDustDataset(Dataset):
    """
    Expect folder structure:
    ROOT_DIR/
        wind_0.47/
            avg_157.6/
                Image_20250714..._159.6.jpg
                ...
        wind_1.50/
            avg_832.1/
                ...
    Each image filename includes the instantaneous concentration as the last numeric token.
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        for wind_folder in sorted(os.listdir(root_dir)):
            wind_path = os.path.join(root_dir, wind_folder)
            if not os.path.isdir(wind_path):
                continue
            wind_speed = self._parse_first_float(wind_folder)
            for avg_folder in sorted(os.listdir(wind_path)):
                avg_path = os.path.join(wind_path, avg_folder)
                if not os.path.isdir(avg_path):
                    continue
                for fname in sorted(os.listdir(avg_path)):
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    img_path = os.path.join(avg_path, fname)
                    instant_conc = self._parse_last_float(fname)
                    if instant_conc is None:
                        continue
                    self.samples.append((img_path, float(instant_conc), float(wind_speed)))
        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check ROOT_DIR and naming format.")
    def _parse_first_float(self, s):
        s2 = s.replace('，', '.').replace(',', '.')
        m = re.search(r'[-+]?\d*\.\d+|\d+', s2)
        return float(m.group(0)) if m else 0.0
    def _parse_last_float(self, s):
        s2 = s.replace('，', '.').replace(',', '.')
        matches = re.findall(r'[-+]?\d*\.\d+|\d+', s2)
        return matches[-1] if matches else None
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, conc, wind = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(conc, dtype=torch.float32), torch.tensor(wind, dtype=torch.float32)

# Transforms
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
train_transform = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)], p=0.5),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# ------------------------ Modules ------------------------
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
    def forward(self, x):
        # x: B,C,H,W
        b, c, h, w = x.size()
        y = x.mean(dim=(-2, -1))  # B,C
        y = y.unsqueeze(1)        # B,1,C
        y = self.conv(y)          # B,1,C
        y = torch.sigmoid(y).squeeze(1)  # B,C
        return x * y.view(b, c, 1, 1)

class LAMP(nn.Module):
    """LAMP: ECA + GAP + GMP + concat"""
    def __init__(self, in_channels):
        super().__init__()
        self.eca = ECA(in_channels, k_size=3)
    def forward(self, x):
        """
        x: B,C,H,W
        returns:
            feature_map_att: B,C,H,W (after ECA)
            F_vec: B,2C (concat GAP & GMP)
        """
        x_att = self.eca(x)
        gap = F.adaptive_avg_pool2d(x_att, 1).view(x_att.size(0), -1)  # B,C
        gmp = F.adaptive_max_pool2d(x_att, 1).view(x_att.size(0), -1)  # B,C
        F_vec = torch.cat([gap, gmp], dim=1)  # B,2C
        return x_att, F_vec

class FiLMBlockVector(nn.Module):
    """
    Single vector-wise FiLM block:
      wind(B,1) -> MLP -> (B,4C) -> chunk -> gamma, beta (B,2C each)
      apply F' = (1+gamma)*F + beta
    """
    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim * 2)  # outputs 2*(2C) in our usage where feat_dim = 2C
        )
        # initialize last layer to zero for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, F_vec, wind):
        """
        F_vec: B, feat_dim (feat_dim = 2C)
        wind: B or Bx1
        """
        if wind.dim() == 1:
            wind = wind.unsqueeze(1)
        params = self.net(wind)  # B, 2*feat_dim
        gamma, beta = params.chunk(2, dim=1)  # each B, feat_dim
        # apply modulation
        Fp = (1.0 + gamma) * F_vec + beta
        return Fp, gamma.detach().cpu(), beta.detach().cpu()

class FiLMStackVector(nn.Module):
    """Stack of N FiLM vector blocks applied sequentially to F"""
    def __init__(self, feat_dim, n_layers=1, hidden=64):
        super().__init__()
        self.blocks = nn.ModuleList([FiLMBlockVector(feat_dim, hidden=hidden) for _ in range(n_layers)])
    def forward(self, F_vec, wind):
        gamma_list = []
        beta_list = []
        out = F_vec
        for b in self.blocks:
            out, g, bt = b(out, wind)
            gamma_list.append(g)
            beta_list.append(bt)
        # return final out and list of intermediates stacked (each B,feat_dim)
        # convert lists to tensors: (n_layers, B, feat_dim)
        gammas = torch.stack(gamma_list, dim=0) if len(gamma_list) > 0 else None
        betas = torch.stack(beta_list, dim=0) if len(beta_list) > 0 else None
        return out, gammas, betas

class FiLMBlockFeatureMap(nn.Module):
    """
    Feature-map-wise FiLM:
      wind(B,1) -> MLP -> outputs (B,2C) -> chunk -> gamma,beta (B,C)
      reshape to (B,C,1,1) and apply X'' = (1+gamma)*X + beta
    """
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, feat_map, wind):
        # feat_map: B,C,H,W
        if wind.dim() == 1:
            wind = wind.unsqueeze(1)
        params = self.net(wind)  # B, 2C
        gamma, beta = params.chunk(2, dim=1)  # each B,C
        gamma = gamma.view(feat_map.size(0), feat_map.size(1), 1, 1)
        beta = beta.view(feat_map.size(0), feat_map.size(1), 1, 1)
        out = (1.0 + gamma) * feat_map + beta
        return out, gamma.detach().cpu(), beta.detach().cpu()

class HeteroscedasticHead(nn.Module):
    """
    Take F' (B, feat_dim) and produce mu (B,) and log_var (B,)
    """
    def __init__(self, feat_dim):
        super().__init__()
        hidden = max(64, feat_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True)
        )
        self.mu_fc = nn.Linear(64, 1)
        self.logvar_fc = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.mu_fc.weight, nonlinearity='linear')
        nn.init.zeros_(self.mu_fc.bias)
        nn.init.kaiming_normal_(self.logvar_fc.weight, nonlinearity='linear')
        nn.init.constant_(self.logvar_fc.bias, -3.0)
    def forward(self, feat):
        h = self.net(feat)
        mu = self.mu_fc(h).squeeze(1)
        logvar = self.logvar_fc(h).squeeze(1)
        return mu, logvar

# ------------------------ Full model ------------------------
class CoalDustModel(nn.Module):
    """
    backbone -> LAMP -> FiLM (vector stack or feature-map) -> Head
    mode:
      - 'vector' : apply FiLMStackVector on pooled vector F (B,2C)
      - 'feature': apply FiLMBlockFeatureMap on feature map X_att (B,C,H,W) before pooling
    num_film_layers: number for vector mode (stack depth)
    """
    def __init__(self, backbone_name=BACKBONE_NAME, pretrained=True, mode='vector', num_film_layers=1):
        super().__init__()
        self.mode = mode
        self.num_film_layers = num_film_layers
        # create backbone
        if USE_TIMM:
            try:
                self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
            except Exception as e:
                raise RuntimeError(f"timm.create_model failed for {backbone_name}: {e}")
            # features_only True returns list of feature maps; we'll use last feature map
        else:
            # torchvision fallback: efficientnet_v2_s
            try:
                bex = tv_models.efficientnet_v2_s(pretrained=pretrained)
                # extract feature extractor: use bex.features which is nn.Sequential
                # wrap into simple callable that returns last feature map
                class TVBackbone(nn.Module):
                    def __init__(self, features):
                        super().__init__()
                        self.features = features
                    def forward(self, x):
                        # return list-like for compatibility: [.., last]
                        outs = []
                        out = x
                        for layer in self.features:
                            out = layer(out)
                        # out is final feature map
                        return [out]
                self.backbone = TVBackbone(bex.features)
            except Exception as e:
                raise RuntimeError("No suitable backbone available. Install timm or use torchvision >= 0.13.")
        # determine out channels C
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feats = self.backbone(dummy)
            last = feats[-1] if isinstance(feats, (list, tuple)) else feats
            self.C = last.shape[1]
        self.train()
        # LAMP
        self.lamp = LAMP(self.C)
        # FiLM
        if mode == 'vector':
            feat_dim = 2 * self.C
            self.film_stack = FiLMStackVector(feat_dim, n_layers=num_film_layers, hidden=128)
            self.head = HeteroscedasticHead(feat_dim)
        elif mode == 'feature':
            self.film_feature = FiLMBlockFeatureMap(self.C, hidden=128)
            # head: after LAMP if feature-wise we still pool and concat to 2C then head
            feat_dim = 2 * self.C
            self.head = HeteroscedasticHead(feat_dim)
        else:
            raise ValueError("mode must be 'vector' or 'feature'")
        # register hooks for simple Grad-CAM: store last feature map for visualization
        self._last_feat = None
        self._last_grad = None
    def forward(self, x, wind):
        feats = self.backbone(x)
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats  # B,C,H,W
        # store for gradcam
        self._last_feat = last
        if self.mode == 'vector':
            # LAMP: get att feature map and pooled vector
            x_att, F_vec = self.lamp(last)  # x_att B,C,H,W ; F_vec B,2C
            Fp, gammas, betas = self.film_stack(F_vec, wind)  # Fp : B,2C
            mu, logvar = self.head(Fp)
            # Return also gammas/betas for analysis (caller can ignore)
            return mu, logvar, gammas, betas
        else:
            # feature-map FiLM
            x_att, F_vec = self.lamp(last)  # x_att B,C,H,W
            x_mod, gamma, beta = self.film_feature(x_att, wind)  # B,C,H,W
            # pool after modulation
            gap = F.adaptive_avg_pool2d(x_mod, 1).view(x_mod.size(0), -1)
            gmp = F.adaptive_max_pool2d(x_mod, 1).view(x_mod.size(0), -1)
            Fp = torch.cat([gap, gmp], dim=1)  # B,2C
            mu, logvar = self.head(Fp)
            return mu, logvar, gamma.unsqueeze(0), beta.unsqueeze(0)

# ------------------------ Loss and metrics ------------------------
def heteroscedastic_loss(mu, log_var, y_true, eps=1e-6):
    # mu, log_var, y_true: tensors shape (B,)
    # clamp log_var for numeric stability
    log_var = torch.clamp(log_var, min=-10.0, max=10.0)
    precision = torch.exp(-log_var)
    loss = 0.5 * (precision * (y_true - mu)**2 + log_var)
    return loss.mean()

def evaluate_metrics(preds, targets):
    preds = np.array(preds, dtype=float)
    targets = np.array(targets, dtype=float)
    n = len(preds)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    # R2:
    denom = np.sum((targets - targets.mean())**2)
    r2 = 1.0 - (np.sum((targets - preds)**2) / (denom + 1e-12))
    # MAPE (use safe denom)
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / (targets[mask] + 1e-8))) * 100 if mask.sum() > 0 else np.nan
    return mae, rmse, r2, mape

# ------------------------ Data split utility ------------------------
def stratified_split_by_bins(dataset, ratio=0.8, seed=SEED):
    rng = random.Random(seed)
    bins = defaultdict(list)
    for i, (_, conc, _) in enumerate(dataset.samples):
        b = int(min(math.floor(conc / 100.0), 9))
        bins[b].append(i)
    train_idx, val_idx = [], []
    for b, idxs in bins.items():
        rng.shuffle(idxs)
        cut = int(len(idxs) * ratio)
        if cut < 1 and len(idxs) >= 1:
            cut = 1
        train_idx += idxs[:cut]
        val_idx += idxs[cut:]
    return train_idx, val_idx

# ------------------------ Training / Validation ------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    for imgs, concs, winds in loader:
        imgs = imgs.to(device)
        concs = concs.to(device)
        winds = winds.to(device)
        optimizer.zero_grad()
        mu, logvar, _, _ = model(imgs, winds)
        loss = heteroscedastic_loss(mu, logvar, concs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_mae += torch.abs(mu - concs).sum().item()
        n += imgs.size(0)
    return total_loss / n, total_mae / n

def validate(model, loader, device, save_gradcam=False, gradcam_out_dir=None):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    preds = []
    targets = []
    gammas_all = []
    betas_all = []
    sample_images = []
    with torch.no_grad():
        for imgs, concs, winds in loader:
            imgs = imgs.to(device)
            concs = concs.to(device)
            winds = winds.to(device)
            mu, logvar, gammas, betas = model(imgs, winds)
            loss = heteroscedastic_loss(mu, logvar, concs)
            total_loss += loss.item() * imgs.size(0)
            total_mae += torch.abs(mu - concs).sum().item()
            n += imgs.size(0)
            preds.extend(mu.detach().cpu().numpy().tolist())
            targets.extend(concs.detach().cpu().numpy().tolist())
            if gammas is not None:
                # gammas: (n_layers, B, feat_dim) or (1,B,C,1,1) depending
                gammas_all.append(gammas.cpu().numpy() if isinstance(gammas, torch.Tensor) else None)
            if betas is not None:
                betas_all.append(betas.cpu().numpy() if isinstance(betas, torch.Tensor) else None)
            # store a few sample images for gradcam demonstration
            if save_gradcam and len(sample_images) < NUM_GRADCAM_SAMPLES:
                sample_images.extend(imgs[:NUM_GRADCAM_SAMPLES].detach().cpu())
    metrics = (total_loss / n, total_mae / n, preds, targets, gammas_all, betas_all, sample_images)
    return metrics

# ------------------------ Grad-CAM (simple) ------------------------
def generate_gradcam(model, input_tensor, target_index=None):
    """
    Very lightweight Grad-CAM:
      - Hook gradients of last feature map stored in model._last_feat
      - compute grads w.r.t mu (mean) output
    Returns heatmap numpy (H x W) normalized 0..1
    """
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(next(model.parameters()).device)  # 1,C,H,W
    # forward with gradient enabled for features
    def forward_and_store(x):
        feats = model.backbone(x)
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats
        return last
    # hook to capture gradients
    activations = None
    grad = None
    # forward pass to get activation
    with torch.set_grad_enabled(True):
        feats = model.backbone(input_tensor)
        last = feats[-1] if isinstance(feats, (list, tuple)) else feats  # 1,C,H,W
        last = last.requires_grad_(True)
        # monkey-patch model._last_feat so that model's forward uses it? simpler: compute mu directly using LAMP+FiLM path
        # Here we'll recompute forward parts manually
        x_att, F_vec = model.lamp(last)
        if model.mode == 'vector':
            Fp, _, _ = model.film_stack(F_vec, torch.zeros(1,1).to(input_tensor.device))  # use zero wind for gradient path
            mu, logvar = model.head(Fp)
        else:
            x_mod, _, _ = model.film_feature(x_att, torch.zeros(1,1).to(input_tensor.device))
            gap = F.adaptive_avg_pool2d(x_mod, 1).view(1, -1)
            gmp = F.adaptive_max_pool2d(x_mod, 1).view(1, -1)
            Fp = torch.cat([gap, gmp], dim=1)
            mu, logvar = model.head(Fp)
        # choose scalar to backprop (mu)
        score = mu if target_index is None else mu[0]
        model.zero_grad()
        score.backward(retain_graph=True)
        # grads on last
        if last.grad is not None:
            grads = last.grad.detach().cpu().numpy()[0]  # C,H,W
        else:
            # try to fetch grad through x_att
            grads = None
        acts = last.detach().cpu().numpy()[0]  # C,H,W
        if grads is None:
            # fallback: approximate using activations only (not ideal)
            cam = np.mean(acts, axis=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        # compute weights
        weights = np.mean(grads, axis=(1,2))  # C
        cam = np.sum(weights[:, None, None] * acts, axis=0)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ------------------------ Experiment runner ------------------------
def run_experiment(mode='vector', film_layers=1, exp_name=None):
    """
    mode: 'vector' or 'feature'
    film_layers: used when mode=='vector' as stack depth
    """
    if exp_name is None:
        exp_name = f"FiLM_{mode}_{film_layers}"
    print("=== Running experiment:", exp_name, "on device", DEVICE)
    # build dataset and splits
    full_dataset = CoalDustDataset(ROOT_DIR, transform=train_transform)
    n_total = len(full_dataset)
    train_idx, val_idx = stratified_split_by_bins(full_dataset, ratio=0.8, seed=SEED)
    train_set = Subset(full_dataset, train_idx)
    val_dataset = CoalDustDataset(ROOT_DIR, transform=val_transform)
    val_set = Subset(val_dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    # model
    model = CoalDustModel(backbone_name=BACKBONE_NAME, pretrained=True, mode=mode, num_film_layers=film_layers)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    # training
    best_val_mae = 1e9
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    out_dir = os.path.join(SAVE_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = validate(model, val_loader, DEVICE, save_gradcam=False)
        val_loss, val_mae, val_preds, val_targets, gammas_all, betas_all, sample_images = val_metrics
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss); history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss); history['val_mae'].append(val_mae)
        epoch_time = time.time() - t0
        print(f"[{exp_name}] Epoch {epoch}/{NUM_EPOCHS}  time={epoch_time:.1f}s  TrainLoss={train_loss:.4f} TrainMAE={train_mae:.3f}  ValLoss={val_loss:.4f} ValMAE={val_mae:.3f}")
        # save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(out_dir, "best_model.pth"))
    # after training evaluate fully and compute metrics
    print("Evaluating final model...")
    val_loss, val_mae, val_preds, val_targets, gammas_all, betas_all, sample_images = validate(model, val_loader, DEVICE, save_gradcam=True)
    mae, rmse, r2, mape = evaluate_metrics(val_preds, val_targets)
    result = {'exp_name': exp_name, 'mode': mode, 'film_layers': film_layers, 'val_loss': val_loss, 'val_mae': val_mae,
              'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'n_samples': len(val_targets)}
    # save metrics and history
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez(os.path.join(out_dir, "history.npz"), **history)
    # plot training curves
    epochs = list(range(1, NUM_EPOCHS+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.legend(); plt.xlabel('epoch'); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_mae'], label='train_mae')
    plt.plot(epochs, history['val_mae'], label='val_mae')
    plt.legend(); plt.xlabel('epoch'); plt.title('MAE')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curves.png"))
    plt.close()
    # gamma/beta hist (if available)
    if gammas_all and len(gammas_all) > 0:
        # gammas_all is list of arrays collected per batch; stack and plot distribution for last layer
        try:
            # concatenate along batch axis
            g_all = np.concatenate([g.reshape(-1, g.shape[-1]) if g is not None else np.zeros((0,1)) for g in gammas_all], axis=0)
            b_all = np.concatenate([b.reshape(-1, b.shape[-1]) if b is not None else np.zeros((0,1)) for b in betas_all], axis=0)
            plt.figure(figsize=(8,3))
            plt.subplot(1,2,1)
            plt.hist(g_all.ravel(), bins=50)
            plt.title('Gamma distribution')
            plt.subplot(1,2,2)
            plt.hist(b_all.ravel(), bins=50)
            plt.title('Beta distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "gamma_beta_hist.png"))
            plt.close()
        except Exception as e:
            print("Warning: gamma/beta hist save failed:", e)
    # Grad-CAM on few samples
    gc_dir = os.path.join(out_dir, "gradcam")
    os.makedirs(gc_dir, exist_ok=True)
    for i, img in enumerate(sample_images[:NUM_GRADCAM_SAMPLES]):
        try:
            cam = generate_gradcam(model, img)
            # overlay and save
            img_np = ((img.permute(1,2,0).numpy()*np.array(imagenet_std) + np.array(imagenet_mean))).clip(0,1)
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(img_np); plt.axis('off'); plt.title('Image')
            plt.subplot(1,2,2)
            plt.imshow(img_np); plt.imshow(cam, cmap='jet', alpha=0.5); plt.axis('off'); plt.title('Grad-CAM')
            plt.tight_layout()
            plt.savefig(os.path.join(gc_dir, f"gradcam_{i}.png"))
            plt.close()
        except Exception as e:
            print("Warning: gradcam failed:", e)
    print("Experiment finished:", exp_name, "results:", result)
    return result

# ------------------------ Main: launch experiments ------------------------
def main():
    experiments = []
    # run vector-wise FiLM for layer counts
    for n in FILM_LAYER_LIST:
        res = run_experiment(mode='vector', film_layers=n, exp_name=f"FiLM_vector_{n}")
        experiments.append(res)
    # run feature-map-wise FiLM if requested
    if RUN_FEATURE_MAP_FILM:
        res = run_experiment(mode='feature', film_layers=1, exp_name="FiLM_featuremap")
        experiments.append(res)
    # save summary table
    summary_path = os.path.join(SAVE_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(experiments, f, indent=2)
    # plot summary MAE
    plt.figure(figsize=(6,4))
    xs = [e['film_layers'] if e['mode']=='vector' else 6 for e in experiments]  # feature-map labelled as 6
    ys = [e['mae'] for e in experiments]
    labels = [f"{e['mode']}_{e['film_layers']}" for e in experiments]
    plt.bar(range(len(experiments)), ys)
    plt.xticks(range(len(experiments)), labels, rotation=45)
    plt.ylabel('MAE (mg/m3)')
    plt.title('Experiment MAE comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "summary_mae.png"))
    plt.close()
    print("All experiments done. Summary saved to", summary_path)

if __name__ == "__main__":
    main()



import os, time, math, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from PIL import Image
import timm
from thop import profile

# ---------------- Dataset ----------------
import re
class CoalDustDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for wind_folder in sorted(os.listdir(root_dir)):
            wind_path = os.path.join(root_dir, wind_folder)
            if not os.path.isdir(wind_path): continue
            wind_speed = self._parse_first_float(wind_folder)
            for avg_folder in sorted(os.listdir(wind_path)):
                avg_path = os.path.join(wind_path, avg_folder)
                if not os.path.isdir(avg_path): continue
                for fname in os.listdir(avg_path):
                    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
                    val = self._parse_last_float(fname)
                    if val is None: continue
                    self.samples.append((os.path.join(avg_path, fname), float(val)))
        self.transform = transform
        print(f"[INFO] Loaded {len(self.samples)} samples")

    def _parse_first_float(self, s):
        s = s.replace(',', '.').replace('，', '.')
        m = re.search(r'\d+\.?\d*', s)
        return float(m.group(0)) if m else 0.0

    def _parse_last_float(self, s):
        s = s.replace(',', '.').replace('，', '.')
        m = re.findall(r'\d+\.?\d*', s)
        return float(m[-1]) if m else None

    def __getitem__(self, idx):
        path, val = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, val

    def __len__(self): return len(self.samples)

# ---------------- Helper Functions ----------------
def get_params_m(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_gflops(model, size=224):
    inp = torch.randn(1, 3, size, size)
    macs, _ = profile(model, inputs=(inp,), verbose=False)
    return macs / 1e9

def measure_fps(model, device, size=224, runs=50):
    model.eval()
    inp = torch.randn(1, 3, size, size).to(device)
    with torch.no_grad():
        # warmup
        for _ in range(10): _ = model(inp)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = model(inp)
        torch.cuda.synchronize()
    avg = (time.time() - t0) / runs
    return 1 / avg

def eval_model(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            out = model(imgs).squeeze().detach().cpu().numpy()
            preds.extend(out)
            gts.extend(targets.numpy())
    mae = mean_absolute_error(gts, preds)
    rmse = math.sqrt(mean_squared_error(gts, preds))
    r2 = r2_score(gts, preds)
    return mae, rmse, r2

def compute_pet(df):
    best_mae = df['MAE'].min()
    perf = best_mae / df['MAE']
    params_n = (df['Params'] - df['Params'].min()) / (df['Params'].max() - df['Params'].min() + 1e-9)
    gflops_n = (df['GFLOPs'] - df['GFLOPs'].min()) / (df['GFLOPs'].max() - df['GFLOPs'].min() + 1e-9)
    cost_n = 0.6 * params_n + 0.4 * gflops_n
    pet = perf / (1 + cost_n)
    df['PET'] = pet
    return df.sort_values('PET', ascending=False).reset_index(drop=True)

# ---------------- Main ----------------
def main():
    DATA_ROOT = "D:\project\A_model\data"  # 修改为你的数据集路径
    IMG_SIZE = 224
    BATCH = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = CoalDustDataset(DATA_ROOT, transform)
    subset = Subset(dataset, range(min(1500, len(dataset))))  # 采样1500张加速测试
    loader = DataLoader(subset, batch_size=BATCH, shuffle=False, num_workers=4)

    models = {
        "EfficientNetV2-S": "tf_efficientnetv2_s",
        "EfficientNetV2-M": "tf_efficientnetv2_m",
        "EfficientNetV2-L": "tf_efficientnetv2_l"
    }

    results = []
    for name, tname in models.items():
        print(f"\n▶ Evaluating {name}")
        model = timm.create_model(tname, pretrained=False, num_classes=1).to(device)

        params = get_params_m(model)
        gflops = get_gflops(model)
        fps = measure_fps(model, device)

        mae, rmse, r2 = eval_model(model, loader, device)

        results.append({
            "Model": name, "MAE": mae, "RMSE": rmse, "R²": r2,
            "Params": params, "GFLOPs": gflops, "FPS": fps
        })

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df = compute_pet(df)
    df.to_csv("efficientnetv2_pet_results.csv", index=False)
    print("\n=== PET Results ===\n", df)

    # Plot
    plt.figure(figsize=(7,4))
    plt.bar(df["Model"], df["PET"], color="#2a9d8f")
    plt.ylabel("PET score (higher = better)")
    plt.title("Performance–Efficiency Tradeoff (EfficientNetV2 series)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("efficientnetv2_pet_plot.png", dpi=300)
    plt.close()
    print("Saved figure to efficientnetv2_pet_plot.png")

if __name__ == "__main__":
    main()

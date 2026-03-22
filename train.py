"""
=============================================================
 train.py — Drowsy Driver CNN Training
 Dataset  : Driver Drowsiness Dataset (DDD)
 Model    : ResNet-18 fine-tuned
 Hardware : RTX 3050, i5, 8GB RAM
=============================================================
 USAGE:
   python train.py
=============================================================
"""

import os, time, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from tqdm import tqdm

# =============================================================
# CONFIG
# =============================================================
DATASET_PATH = r"C:\Users\param\Downloads\DrowsyDriverDataset2\Driver Drowsiness Dataset (DDD)"
RESULTS_DIR  = "./results"
MODELS_DIR   = "./models"
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-4
NUM_WORKERS  = 4
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# =============================================================
# DATA
# =============================================================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

print("\n[1/4] Loading dataset...")
full_ds     = datasets.ImageFolder(DATASET_PATH)
class_names = full_ds.classes
print(f"  Classes : {class_names}")
print(f"  Total   : {len(full_ds):,} images")

n       = len(full_ds)
n_train = int(0.80 * n)
n_val   = int(0.10 * n)
n_test  = n - n_train - n_val

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)
train_ds.dataset.transform = train_tf
val_ds.dataset.transform   = val_tf
test_ds.dataset.transform  = val_tf
print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

# Weighted sampler for class imbalance
targets     = [full_ds.targets[i] for i in train_ds.indices]
cls_counts  = np.bincount(targets)
wts         = 1.0 / cls_counts
sample_wts  = [wts[t] for t in targets]
sampler     = WeightedRandomSampler(sample_wts, len(sample_wts))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# =============================================================
# MODEL
# =============================================================
print("\n[2/4] Building model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {device}")
if device.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze early layers, fine-tune layer3, layer4, fc
for name, param in model.named_parameters():
    param.requires_grad = any(k in name for k in ["layer3","layer4","fc"])

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
model = model.to(device)

total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params     : {total_p:,}")
print(f"  Trainable params : {trainable_p:,}")

# =============================================================
# TRAINING
# =============================================================
print("\n[3/4] Training...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_val   = 0.0
best_path  = os.path.join(MODELS_DIR, "best_model.pth")
history    = {"tl":[], "vl":[], "ta":[], "va":[]}
start      = time.time()

for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    rl, rc, rt = 0.0, 0, 0
    for imgs, lbls in tqdm(train_loader, desc=f"Ep {epoch:02d}/{EPOCHS} Train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        rl += loss.item()*imgs.size(0)
        _, p = torch.max(out,1)
        rc += (p==lbls).sum().item(); rt += lbls.size(0)
    tl = rl/rt; ta = rc/rt

    # Validate
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out  = model(imgs)
            loss = criterion(out, lbls)
            vl += loss.item()*imgs.size(0)
            _, p = torch.max(out,1)
            vc += (p==lbls).sum().item(); vt += lbls.size(0)
    vl = vl/vt; va = vc/vt
    scheduler.step()

    history["tl"].append(tl); history["vl"].append(vl)
    history["ta"].append(ta); history["va"].append(va)
    print(f"  Epoch {epoch:02d} | Train Loss {tl:.4f} Acc {ta*100:.2f}% | Val Loss {vl:.4f} Acc {va*100:.2f}%")

    if va > best_val:
        best_val = va
        torch.save(model.state_dict(), best_path)
        print(f"           ✓ Best saved ({best_val*100:.2f}%)")

train_time = time.time() - start
print(f"\n  Training complete in {train_time/60:.1f} minutes")

# =============================================================
# EVALUATION
# =============================================================
print("\n[4/4] Evaluating on test set...")
model.load_state_dict(torch.load(best_path))
model.eval()

preds, labels, probs, inf_times = [], [], [], []
with torch.no_grad():
    for imgs, lbls in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        t0   = time.time()
        out  = model(imgs)
        inf_times.append((time.time()-t0)/imgs.size(0)*1000)
        pr   = torch.softmax(out,1)[:,1].cpu().numpy()
        _, p = torch.max(out,1)
        preds.extend(p.cpu().numpy())
        labels.extend(lbls.numpy())
        probs.extend(pr)

preds  = np.array(preds)
labels = np.array(labels)
probs  = np.array(probs)

acc  = accuracy_score(labels, preds)
prec = precision_score(labels, preds, average="weighted")
rec  = recall_score(labels, preds, average="weighted")
f1   = f1_score(labels, preds, average="weighted")
auc  = roc_auc_score(labels, probs)
fps  = 1000 / np.mean(inf_times)

print(f"\n  ┌─────────────────────────────────┐")
print(f"  │  FINAL TEST RESULTS             │")
print(f"  ├─────────────────────────────────┤")
print(f"  │  Accuracy   : {acc*100:6.2f}%           │")
print(f"  │  Precision  : {prec*100:6.2f}%           │")
print(f"  │  Recall     : {rec*100:6.2f}%           │")
print(f"  │  F1 Score   : {f1*100:6.2f}%           │")
print(f"  │  ROC-AUC    : {auc:.4f}             │")
print(f"  │  Est. FPS   : {fps:6.1f}             │")
print(f"  └─────────────────────────────────┘")
print(f"\n{classification_report(labels, preds, target_names=class_names)}")

# =============================================================
# PLOTS
# =============================================================
ep = range(1, EPOCHS+1)

# Training curves
fig, ax = plt.subplots(1,2,figsize=(14,5))
ax[0].plot(ep,[a*100 for a in history["ta"]],"b-o",label="Train")
ax[0].plot(ep,[a*100 for a in history["va"]],"r-o",label="Val")
ax[0].set_title("Accuracy per Epoch",fontsize=13,fontweight="bold")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy (%)")
ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_ylim([50,102])
ax[1].plot(ep,history["tl"],"b-o",label="Train")
ax[1].plot(ep,history["vl"],"r-o",label="Val")
ax[1].set_title("Loss per Epoch",fontsize=13,fontweight="bold")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png",dpi=150,bbox_inches="tight")
plt.close(); print("✓ Saved training_curves.png")

# Confusion matrix
cm = confusion_matrix(labels, preds)
fig,ax = plt.subplots(figsize=(7,6))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=class_names,yticklabels=class_names,annot_kws={"size":16})
ax.set_title("Confusion Matrix",fontsize=13,fontweight="bold")
ax.set_ylabel("True"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png",dpi=150,bbox_inches="tight")
plt.close(); print("✓ Saved confusion_matrix.png")

# ROC curve
fpr,tpr,_ = roc_curve(labels,probs)
fig,ax = plt.subplots(figsize=(7,6))
ax.plot(fpr,tpr,"b-",lw=2,label=f"AUC = {auc:.4f}")
ax.plot([0,1],[0,1],"r--",lw=1,label="Random")
ax.set_title("ROC Curve",fontsize=13,fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_curve.png",dpi=150,bbox_inches="tight")
plt.close(); print("✓ Saved roc_curve.png")

# Summary text
with open(f"{RESULTS_DIR}/results_summary.txt","w") as f:
    f.write("="*55+"\n")
    f.write("DROWSY DRIVER DETECTION — RESULTS SUMMARY\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("="*55+"\n\n")
    f.write(f"Dataset\n")
    f.write(f"  Drowsy      : 22,348 images\n")
    f.write(f"  Non-Drowsy  : 19,445 images\n")
    f.write(f"  Total       : 41,793 images\n")
    f.write(f"  Split       : 80/10/10\n\n")
    f.write(f"Model         : ResNet-18 (fine-tuned)\n")
    f.write(f"Epochs        : {EPOCHS}\n")
    f.write(f"Batch Size    : {BATCH_SIZE}\n")
    f.write(f"Training Time : {train_time/60:.1f} minutes\n\n")
    f.write(f"Test Results\n")
    f.write(f"  Accuracy    : {acc*100:.2f}%\n")
    f.write(f"  Precision   : {prec*100:.2f}%\n")
    f.write(f"  Recall      : {rec*100:.2f}%\n")
    f.write(f"  F1 Score    : {f1*100:.2f}%\n")
    f.write(f"  ROC-AUC     : {auc:.4f}\n")
    f.write(f"  FPS         : {fps:.1f}\n\n")
    f.write(classification_report(labels, preds, target_names=class_names))

print("✓ Saved results_summary.txt")
print("\n✅ ALL DONE — check the ./results/ folder")
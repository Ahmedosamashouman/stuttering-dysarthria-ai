from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


FEATURE_DIR = Path("data/processed/features")

OUT_MODEL = Path("outputs/models/cnn_bilstm_mfcc_fusion_v3.pt")
OUT_METRICS = Path("outputs/metrics/cnn_bilstm_mfcc_fusion_v3_metrics.json")
OUT_HISTORY = Path("outputs/metrics/cnn_bilstm_mfcc_fusion_v3_history.csv")
OUT_THRESHOLDS = Path("outputs/metrics/cnn_bilstm_mfcc_fusion_v3_thresholds.csv")
OUT_PREDS = Path("outputs/predictions/cnn_bilstm_mfcc_fusion_v3_test_predictions.csv")

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
SEED = 42

N_MELS = 64
TIME_STEPS = 301
MFCC_DIM = 480


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FusionDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        path = FEATURE_DIR / f"{split}_features.npz"

        if not path.exists():
            raise FileNotFoundError(f"Missing feature file: {path}")

        data = np.load(path)

        self.logmel = data["logmel"].astype(np.float32)
        self.mfcc_stats = data["mfcc_stats"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        self.paths = data["paths"]
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _spec_augment(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: 64 x 301

        Stronger but still safe SpecAugment:
        - 2 possible frequency masks
        - 2 possible time masks
        - small Gaussian noise
        """
        x = x.copy()

        # Frequency masking
        for _ in range(2):
            if np.random.rand() < 0.60:
                f = np.random.randint(0, 12)
                if f > 0:
                    f0 = np.random.randint(0, max(1, x.shape[0] - f))
                    x[f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(2):
            if np.random.rand() < 0.60:
                t = np.random.randint(0, 40)
                if t > 0:
                    t0 = np.random.randint(0, max(1, x.shape[1] - t))
                    x[:, t0:t0 + t] = 0.0

        # Small noise
        if np.random.rand() < 0.30:
            noise = np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)
            x = x + noise

        return x.astype(np.float32)

    def __getitem__(self, idx):
        x_logmel = self.logmel[idx]
        x_mfcc = self.mfcc_stats[idx]
        y = self.labels[idx]
        path = str(self.paths[idx])

        if self.augment:
            x_logmel = self._spec_augment(x_logmel)

        x_logmel = torch.tensor(x_logmel, dtype=torch.float32).unsqueeze(0)
        x_mfcc = torch.tensor(x_mfcc, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x_logmel, x_mfcc, y, path


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: batch x time x hidden
        weights = torch.softmax(self.attention(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled


class CNNBiLSTMMFCCFusionV3(nn.Module):
    """
    V3 model:
    - Log-mel branch: CNN -> BiLSTM -> Attention
    - MFCC branch: MLP over 480-dim MFCC stats
    - Fusion classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 128,
        dropout: float = 0.40,
        mfcc_dim: int = 480,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Dropout2d(dropout),
        )

        # Mel bins: 64 -> 32 -> 16 -> 8
        # Channels: 96
        # LSTM input = 96 * 8 = 768
        self.lstm = nn.LSTM(
            input_size=96 * 8,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = AttentionPooling(hidden_size * 2)

        # MFCC statistics branch
        self.mfcc_branch = nn.Sequential(
            nn.LayerNorm(mfcc_dim),
            nn.Linear(mfcc_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # BiLSTM output = hidden_size * 2 = 256
        # MFCC branch output = 64
        # Fusion = 320
        fusion_dim = hidden_size * 2 + 64

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_logmel, x_mfcc):
        # x_logmel: batch x 1 x 64 x 301
        z = self.cnn(x_logmel)

        # z: batch x channels x freq x time
        b, c, f, t = z.shape

        # batch x time x features
        z = z.permute(0, 3, 1, 2).contiguous()
        z = z.view(b, t, c * f)

        z, _ = self.lstm(z)
        z = self.attention(z)

        m = self.mfcc_branch(x_mfcc)

        fused = torch.cat([z, m], dim=1)
        logits = self.classifier(fused)

        return logits


class FocalLoss(nn.Module):
    """
    Focal loss focuses training on hard examples.

    Useful here because fluent clips are minority and many fluent examples are
    confused with stutter.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )

        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce

        return focal.mean()


def compute_metrics_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))

    precision_stutter = tp / max(1, tp + fp)
    recall_stutter = tp / max(1, tp + fn)
    f1_stutter = 2 * precision_stutter * recall_stutter / max(
        1e-12,
        precision_stutter + recall_stutter,
    )

    precision_fluent = tn / max(1, tn + fn)
    specificity_fluent = tn / max(1, tn + fp)
    f1_fluent = 2 * precision_fluent * specificity_fluent / max(
        1e-12,
        precision_fluent + specificity_fluent,
    )

    macro_f1 = (f1_stutter + f1_fluent) / 2
    balanced_accuracy = (recall_stutter + specificity_fluent) / 2
    youden_j = recall_stutter + specificity_fluent - 1.0

    return {
        "accuracy": float(accuracy),
        "precision_stutter": float(precision_stutter),
        "recall_stutter_sensitivity": float(recall_stutter),
        "specificity_fluent": float(specificity_fluent),
        "f1_stutter": float(f1_stutter),
        "f1_fluent": float(f1_fluent),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "youden_j": float(youden_j),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def preds_from_probs(prob_stutter, threshold):
    prob_stutter = np.asarray(prob_stutter)
    return (prob_stutter >= threshold).astype(np.int64)


def tune_threshold(y_true, prob_stutter):
    """
    We save all thresholds, but select by macro-F1.

    After training, you can also inspect:
    - best Youden J
    - best specificity with recall >= 75%
    """
    thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []
    best = None

    for threshold in thresholds:
        y_pred = preds_from_probs(prob_stutter, threshold)
        metrics = compute_metrics_from_arrays(y_true, y_pred)

        row = {
            "threshold": float(threshold),
            **metrics,
        }

        rows.append(row)

        if best is None or row["macro_f1"] > best["macro_f1"]:
            best = row

    OUT_THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)

    with OUT_THRESHOLDS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return best, rows


def build_balanced_sampler(labels):
    labels = np.asarray(labels).astype(np.int64)
    class_counts = np.bincount(labels, minlength=2)

    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, class_counts


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_true = []
    all_prob = []

    for x_logmel, x_mfcc, y, _ in loader:
        x_logmel = x_logmel.to(device)
        x_mfcc = x_mfcc.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(x_logmel, x_mfcc)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]

        total_loss += loss.item() * x_logmel.size(0)
        all_true.extend(y.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

    all_true = np.asarray(all_true)
    all_prob = np.asarray(all_prob)
    all_pred = preds_from_probs(all_prob, threshold=0.5)

    metrics = compute_metrics_from_arrays(all_true, all_pred)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))

    return metrics, all_true, all_prob


def evaluate_with_paths(model, loader, criterion, device, threshold):
    model.eval()

    total_loss = 0.0
    all_paths = []
    all_true = []
    all_prob = []

    with torch.no_grad():
        for x_logmel, x_mfcc, y, paths in loader:
            x_logmel = x_logmel.to(device)
            x_mfcc = x_mfcc.to(device)
            y = y.to(device)

            logits = model(x_logmel, x_mfcc)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * x_logmel.size(0)

            all_paths.extend(paths)
            all_true.extend(y.cpu().numpy().tolist())
            all_prob.extend(probs.cpu().numpy().tolist())

    all_true = np.asarray(all_true)
    all_prob = np.asarray(all_prob)
    all_pred = preds_from_probs(all_prob, threshold=threshold)

    metrics = compute_metrics_from_arrays(all_true, all_pred)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    metrics["threshold"] = float(threshold)

    return metrics, all_paths, all_true, all_pred, all_prob


def save_history(history):
    OUT_HISTORY.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    with OUT_HISTORY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_predictions(paths, y_true, y_pred, prob_stutter):
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PREDS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "y_true", "y_pred", "prob_stutter"])

        for p, yt, yp, prob in zip(paths, y_true, y_pred, prob_stutter):
            writer.writerow([p, int(yt), int(yp), float(prob)])


def main():
    set_seed(SEED)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    train_ds = FusionDataset("train", augment=True)
    val_ds = FusionDataset("val", augment=False)
    test_ds = FusionDataset("test", augment=False)

    sampler, class_counts = build_balanced_sampler(train_ds.labels)

    print("[INFO] Dataset sizes:")
    print("train:", len(train_ds))
    print("val:", len(val_ds))
    print("test:", len(test_ds))
    print("[INFO] Class counts:", class_counts.tolist())
    print("[INFO] Features:")
    print("logmel:", train_ds.logmel.shape)
    print("mfcc_stats:", train_ds.mfcc_stats.shape)
    print("[INFO] V3 active: logmel + MFCC fusion + focal loss + stronger SpecAugment")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    model = CNNBiLSTMMFCCFusionV3(
        num_classes=2,
        hidden_size=128,
        dropout=0.40,
        mfcc_dim=MFCC_DIM,
    ).to(device)

    criterion = FocalLoss(gamma=2.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_macro_f1 = -1.0
    patience_left = PATIENCE
    history = []

    print("[INFO] Start v3 training...")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
        )

        val_metrics_05, val_true, val_prob = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            train_mode=False,
        )

        best_threshold_row, _ = tune_threshold(val_true, val_prob)

        tuned_threshold = float(best_threshold_row["threshold"])
        tuned_val_macro_f1 = float(best_threshold_row["macro_f1"])

        scheduler.step(tuned_val_macro_f1)

        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy_05": train_metrics["accuracy"],
            "train_macro_f1_05": train_metrics["macro_f1"],
            "val_loss": val_metrics_05["loss"],
            "val_accuracy_05": val_metrics_05["accuracy"],
            "val_macro_f1_05": val_metrics_05["macro_f1"],
            "val_best_threshold": tuned_threshold,
            "val_best_macro_f1": tuned_val_macro_f1,
            "val_best_specificity_fluent": best_threshold_row["specificity_fluent"],
            "val_best_recall_stutter": best_threshold_row["recall_stutter_sensitivity"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }

        history.append(row)
        save_history(history)

        print(
            f"[EPOCH {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_macro_f1_05={train_metrics['macro_f1']:.4f} "
            f"val_macro_f1_05={val_metrics_05['macro_f1']:.4f} "
            f"val_best_macro_f1={tuned_val_macro_f1:.4f} "
            f"threshold={tuned_threshold:.2f} "
            f"time={elapsed:.1f}s"
        )

        if tuned_val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = tuned_val_macro_f1
            patience_left = PATIENCE

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_threshold": tuned_threshold,
                    "class_counts": class_counts.tolist(),
                    "architecture": "CNNBiLSTMMFCCFusionV3",
                    "input_logmel_shape": [1, N_MELS, TIME_STEPS],
                    "input_mfcc_dim": MFCC_DIM,
                },
                OUT_MODEL,
            )

            print(f"[OK] Saved best v3 model: {OUT_MODEL}")
        else:
            patience_left -= 1
            print(f"[INFO] Patience left: {patience_left}")

            if patience_left <= 0:
                print("[STOP] Early stopping")
                break

    print("[INFO] Loading best v3 model for final evaluation...")

    checkpoint = torch.load(OUT_MODEL, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_threshold = float(checkpoint["best_threshold"])

    val_metrics, _, _, _, _ = evaluate_with_paths(
        model,
        val_loader,
        criterion,
        device,
        threshold=best_threshold,
    )

    test_metrics, test_paths, test_true, test_pred, test_prob = evaluate_with_paths(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_threshold,
    )

    save_predictions(test_paths, test_true, test_pred, test_prob)

    final_metrics = {
        "model": "cnn_bilstm_mfcc_fusion_v3",
        "features": "log-mel spectrogram + MFCC statistics fusion",
        "training": "balanced sampler + focal loss + stronger SpecAugment",
        "best_epoch": int(checkpoint["epoch"]),
        "best_threshold": best_threshold,
        "best_val_macro_f1_during_training": float(checkpoint["best_val_macro_f1"]),
        "val": val_metrics,
        "test": test_metrics,
    }

    with OUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n[OK] Saved model:", OUT_MODEL)
    print("[OK] Saved metrics:", OUT_METRICS)
    print("[OK] Saved history:", OUT_HISTORY)
    print("[OK] Saved threshold search:", OUT_THRESHOLDS)
    print("[OK] Saved predictions:", OUT_PREDS)

    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
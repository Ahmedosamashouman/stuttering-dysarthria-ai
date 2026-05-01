"""
M8 — Deep Model 2: Wav2Vec2 SSL Transformer Fine-Tuning (Full Attention).

Strategy:
  - facebook/wav2vec2-base backbone
  - Freeze CNN feature extractor (stable low-level features)
  - Unfreeze top 4 transformer encoder layers (8–11) for task-specific tuning
  - Layer-wise learning rate decay (LLRD): lower LR for earlier layers
  - Linear warmup + cosine annealing schedule
  - Mixed precision (AMP) for speed and memory
  - Waveform-level augmentation: additive noise + time shift
  - Balanced sampling: all fluent + equal stutter (max data)
  - Threshold tuning via Youden J (maximises sensitivity + specificity)
  - Full evaluation: accuracy, precision, recall, specificity, F1, macro-F1,
    balanced accuracy, AUC-ROC (trapezoidal, no sklearn)

No HuggingFace datasets/evaluate. No pandas. No sklearn.
"""

from pathlib import Path
import csv
import json
import random
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import librosa
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Config,
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MANIFEST    = Path("data/processed/manifest.csv")
OUT_MODEL   = Path("outputs/models/wav2vec2_full_attention_test.pt")
OUT_METRICS = Path("outputs/metrics/wav2vec2_full_attention_test_metrics.json")
OUT_HISTORY = Path("outputs/metrics/wav2vec2_full_attention_test_history.csv")
OUT_PREDS   = Path("outputs/predictions/wav2vec2_full_attention_test_predictions.csv")
OUT_THRESH  = Path("outputs/metrics/wav2vec2_full_attention_test_threshold_analysis.csv")

# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
MODEL_NAME       = "facebook/wav2vec2-base"
SAMPLE_RATE      = 16000
TARGET_LEN       = 48000          # 3 s × 16 kHz

# Use ALL fluent train samples + equal stutter for maximum data.
# Val/test stay balanced and capped so evaluation is fair.
TRAIN_PER_CLASS  = 3178           # all fluent in train split
VAL_PER_CLASS    = 800
TEST_PER_CLASS   = 786

BATCH_SIZE        = 1             # lower to 1 on OOM
GRAD_ACCUM_STEPS  = 16             # effective batch = 16
EPOCHS            = 10
WARMUP_RATIO      = 0.10          # 10% of total steps for warmup
PEAK_LR           = 3e-5          # for top transformer layers
LLRD_FACTOR       = 0.85          # each lower layer gets LR × LLRD_FACTOR
WEIGHT_DECAY      = 1e-2
PATIENCE          = 4
SEED              = 42

# Unfreezing: keep feature encoder frozen, unfreeze top N transformer layers.
# wav2vec2-base has 12 transformer encoder layers (0–11).
UNFREEZE_TOP_N_LAYERS = 4         # layers 8, 9, 10, 11

# Augmentation (applied to waveform during training)
AUG_NOISE_PROB   = 0.40
AUG_NOISE_STD    = 0.005
AUG_SHIFT_PROB   = 0.40
AUG_SHIFT_MAX_MS = 200            # up to 200 ms shift


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────
def load_manifest(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def split_by_class(rows, split_name):
    fluent, stutter = [], []
    for r in rows:
        if r["split"] != split_name:
            continue
        if int(r["label_id"]) == 0:
            fluent.append(r)
        else:
            stutter.append(r)
    return fluent, stutter


def balanced_sample(fluent, stutter, n_per_class, rng):
    f = rng.sample(fluent,  min(n_per_class, len(fluent)))
    s = rng.sample(stutter, min(n_per_class, len(stutter)))
    combined = f + s
    rng.shuffle(combined)
    return combined


# ─────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) >= TARGET_LEN:
        y = y[:TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode="constant")
    peak = np.abs(y).max()
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


def augment_waveform(y: np.ndarray) -> np.ndarray:
    """Light waveform augmentation applied only during training."""
    y = y.copy()

    # Additive Gaussian noise
    if random.random() < AUG_NOISE_PROB:
        y += np.random.normal(0.0, AUG_NOISE_STD, size=y.shape).astype(np.float32)
        y = np.clip(y, -1.0, 1.0)

    # Random time shift (circular)
    if random.random() < AUG_SHIFT_PROB:
        max_shift = int(SAMPLE_RATE * AUG_SHIFT_MAX_MS / 1000)
        shift = random.randint(-max_shift, max_shift)
        y = np.roll(y, shift)

    return y


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, rows, feature_extractor, augment: bool = False):
        self.rows   = rows
        self.fe     = feature_extractor
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r   = self.rows[idx]
        wav = load_audio(r["path"])

        if self.augment:
            wav = augment_waveform(wav)

        enc = self.fe(
            wav,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = enc.input_values.squeeze(0)
        label        = torch.tensor(int(r["label_id"]), dtype=torch.long)
        return input_values, label, r["path"]


def collate_fn(batch):
    vals, labels, paths = zip(*batch)
    return torch.stack(vals, 0), torch.stack(labels, 0), list(paths)


# ─────────────────────────────────────────────
# Model setup: selective unfreezing + LLRD
# ─────────────────────────────────────────────
def build_model():
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    # 1. Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # 2. Always unfreeze classifier projection head
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True

    # 3. Unfreeze top N transformer encoder layers
    num_layers = len(model.wav2vec2.encoder.layers)   # 12 for base
    unfreeze_from = num_layers - UNFREEZE_TOP_N_LAYERS

    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        if i >= unfreeze_from:
            for p in layer.parameters():
                p.requires_grad = True

    # 4. Always unfreeze the layer norm after encoder
    for p in model.wav2vec2.encoder.layer_norm.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Params: total={total:,}  trainable={trainable:,}  "
          f"({100*trainable/total:.1f}%)")
    return model


def build_optimizer(model) -> torch.optim.AdamW:
    """
    Layer-wise learning rate decay.
    Classifier head → PEAK_LR
    Each encoder layer going down → × LLRD_FACTOR
    """
    num_layers   = len(model.wav2vec2.encoder.layers)
    unfreeze_from = num_layers - UNFREEZE_TOP_N_LAYERS

    param_groups = []

    # Classifier + projector (highest LR)
    param_groups.append({
        "params": list(model.classifier.parameters()) +
                  list(model.projector.parameters()),
        "lr": PEAK_LR,
        "name": "classifier",
    })

    # Layer norm after encoder
    param_groups.append({
        "params": list(model.wav2vec2.encoder.layer_norm.parameters()),
        "lr": PEAK_LR * (LLRD_FACTOR ** 1),
        "name": "enc_layer_norm",
    })

    # Top transformer layers (layer 11 → highest, layer 8 → lowest of unfrozen)
    for i in range(num_layers - 1, unfreeze_from - 1, -1):
        depth = num_layers - 1 - i          # 0 for top layer
        lr_i  = PEAK_LR * (LLRD_FACTOR ** (depth + 2))
        param_groups.append({
            "params": list(model.wav2vec2.encoder.layers[i].parameters()),
            "lr": lr_i,
            "name": f"encoder_layer_{i}",
        })
        print(f"  [LLRD] encoder layer {i:2d} → lr={lr_i:.2e}")

    for g in param_groups:
        g["weight_decay"] = WEIGHT_DECAY

    return torch.optim.AdamW(param_groups)


# ─────────────────────────────────────────────
# Cosine schedule with linear warmup
# ─────────────────────────────────────────────
def get_scheduler(optimizer, total_steps: int):
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# Metrics (no sklearn)
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc  = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1_s = 2 * prec * rec  / max(1e-12, prec + rec)
    p0   = tn / max(1, tn + fn)
    f1_f = 2 * p0  * spec  / max(1e-12, p0  + spec)
    mf1  = (f1_s + f1_f) / 2
    bal  = (rec  + spec)  / 2
    yj   = rec + spec - 1.0

    return dict(
        accuracy              = float(acc),
        balanced_accuracy     = float(bal),
        macro_f1              = float(mf1),
        precision_stutter     = float(prec),
        recall_stutter        = float(rec),
        specificity_fluent    = float(spec),
        f1_stutter            = float(f1_s),
        f1_fluent             = float(f1_f),
        youden_j              = float(yj),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def auc_roc(y_true, prob_pos) -> float:
    """Manual trapezoidal AUC-ROC. No sklearn."""
    y_true    = np.asarray(y_true, dtype=np.int64)
    prob_pos  = np.asarray(prob_pos, dtype=np.float64)
    thresholds = np.sort(np.unique(prob_pos))[::-1]

    tprs, fprs = [0.0], [0.0]
    P = y_true.sum()
    N = len(y_true) - P

    for thr in thresholds:
        pred = (prob_pos >= thr).astype(np.int64)
        tp   = int(((y_true == 1) & (pred == 1)).sum())
        fp   = int(((y_true == 0) & (pred == 1)).sum())
        tprs.append(tp / max(1, P))
        fprs.append(fp / max(1, N))

    tprs.append(1.0)
    fprs.append(1.0)

    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2.0
    return float(abs(auc))


def tune_threshold(y_true, prob_stutter):
    """Returns best threshold (Youden J) and its metrics."""
    best_j, best_thr, best_m = -1.0, 0.5, None
    rows = []

    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        preds = (prob_stutter >= thr).astype(np.int64)
        m     = compute_metrics(y_true, preds)
        m["threshold"] = float(thr)
        rows.append(m)
        if m["youden_j"] > best_j:
            best_j, best_thr, best_m = m["youden_j"], float(thr), m

    return best_thr, best_m, rows


# ─────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────
def run_train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    all_true, all_prob = [], []
    n_samples = 0

    for step, (xv, y, _) in enumerate(loader, 1):
        xv = xv.to(device, non_blocking=True)
        y  = y.to(device,  non_blocking=True)

        with autocast(enabled=(device == "cuda")):
            out  = model(input_values=xv, labels=y)
            loss = out.loss / grad_accum

        scaler.scale(loss).backward()

        if step % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        probs = torch.softmax(out.logits.detach().float(), dim=1)[:, 1]
        total_loss += out.loss.item() * xv.size(0)
        n_samples  += xv.size(0)
        all_true.extend(y.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

    # Flush remainder only if the last batch did not already step.
    if len(loader) % grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    all_true = np.asarray(all_true)
    all_prob = np.asarray(all_prob)
    preds    = (all_prob >= 0.5).astype(np.int64)
    m        = compute_metrics(all_true, preds)
    m["loss"]    = total_loss / max(1, n_samples)
    m["auc_roc"] = auc_roc(all_true, all_prob)
    return m, all_true, all_prob


@torch.no_grad()
def run_eval_epoch(model, loader, device):
    model.eval()

    total_loss = 0.0
    all_true, all_prob, all_paths = [], [], []
    n_samples = 0

    for xv, y, paths in loader:
        xv = xv.to(device, non_blocking=True)
        y  = y.to(device,  non_blocking=True)

        with autocast(enabled=(device == "cuda")):
            out = model(input_values=xv, labels=y)

        probs = torch.softmax(out.logits.float(), dim=1)[:, 1]
        total_loss += out.loss.item() * xv.size(0)
        n_samples  += xv.size(0)
        all_true.extend(y.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())
        all_paths.extend(paths)

    all_true = np.asarray(all_true)
    all_prob = np.asarray(all_prob)
    return total_loss / max(1, n_samples), all_true, all_prob, all_paths


# ─────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────
def save_history(history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def save_predictions(paths, y_true, y_pred, prob_stutter, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "y_true", "y_pred", "prob_stutter"])
        for p, yt, yp, ps in zip(paths, y_true, y_pred, prob_stutter):
            w.writerow([p, int(yt), int(yp), float(ps)])


def save_threshold_analysis(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    set_seed(SEED)
    rng = random.Random(SEED)

    for p in [OUT_MODEL, OUT_METRICS, OUT_HISTORY, OUT_PREDS, OUT_THRESH]:
        p.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # ── Load manifest ──────────────────────────────────────────
    print("[INFO] Loading manifest...")
    rows = load_manifest(MANIFEST)

    train_rows = balanced_sample(*split_by_class(rows, "train"), TRAIN_PER_CLASS, rng)
    val_rows   = balanced_sample(*split_by_class(rows, "val"),   VAL_PER_CLASS,   rng)
    test_rows  = balanced_sample(*split_by_class(rows, "test"),  TEST_PER_CLASS,  rng)

    train_labels = [int(r["label_id"]) for r in train_rows]
    counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    print(f"[INFO] Train {len(train_rows)} (fluent={counts[0]} stutter={counts[1]})  "
          f"Val {len(val_rows)}  Test {len(test_rows)}")

    # ── Feature extractor + model ──────────────────────────────
    print(f"[INFO] Loading {MODEL_NAME} ...")
    fe    = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = build_model()
    model = model.to(device)

    # ── DataLoaders ────────────────────────────────────────────
    kw = dict(collate_fn=collate_fn, num_workers=0, pin_memory=False)

    train_loader = DataLoader(
        SpeechDataset(train_rows, fe, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, **kw
    )
    val_loader = DataLoader(
        SpeechDataset(val_rows, fe, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, **kw
    )
    test_loader = DataLoader(
        SpeechDataset(test_rows, fe, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, **kw
    )

    # ── Optimizer + scheduler + scaler ────────────────────────
    optimizer    = build_optimizer(model)
    total_steps  = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler    = get_scheduler(optimizer, total_steps)
    scaler       = GradScaler(enabled=(device == "cuda"))

    print(f"[INFO] Total optimizer steps: {total_steps}  "
          f"Warmup: {int(total_steps * WARMUP_RATIO)}")

    # ── Training loop ──────────────────────────────────────────
    best_val_mf1   = -1.0
    best_threshold = 0.5
    patience_left  = PATIENCE
    history        = []

    print("[INFO] Starting training...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_m, _, _ = run_train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, GRAD_ACCUM_STEPS
        )
        va_loss, va_true, va_prob, _ = run_eval_epoch(model, val_loader, device)

        # Threshold tuning on validation (Youden J)
        thr, thr_m, _ = tune_threshold(va_true, va_prob)
        tuned_mf1      = thr_m["macro_f1"]
        va_auc         = auc_roc(va_true, va_prob)

        elapsed = time.time() - t0
        row = dict(
            epoch                 = epoch,
            train_loss            = round(tr_m["loss"],         4),
            train_macro_f1        = round(tr_m["macro_f1"],     4),
            train_auc_roc         = round(tr_m["auc_roc"],      4),
            val_loss              = round(va_loss,               4),
            val_macro_f1_05       = round(
                compute_metrics(va_true, (va_prob >= 0.5).astype(np.int64))["macro_f1"], 4
            ),
            val_macro_f1_tuned    = round(tuned_mf1,            4),
            val_auc_roc           = round(va_auc,               4),
            val_balanced_accuracy = round(thr_m["balanced_accuracy"], 4),
            val_best_threshold    = thr,
            val_recall_stutter    = round(thr_m["recall_stutter"],    4),
            val_specificity_fluent= round(thr_m["specificity_fluent"],4),
            val_youden_j          = round(thr_m["youden_j"],    4),
            lr_classifier         = optimizer.param_groups[0]["lr"],
            seconds               = round(elapsed, 1),
        )
        history.append(row)
        save_history(history, OUT_HISTORY)

        print(
            f"[EPOCH {epoch:02d}/{EPOCHS}] "
            f"tr_loss={tr_m['loss']:.4f} tr_mf1={tr_m['macro_f1']:.4f} "
            f"tr_auc={tr_m['auc_roc']:.4f} | "
            f"va_loss={va_loss:.4f} va_mf1={tuned_mf1:.4f} "
            f"va_auc={va_auc:.4f} va_bal={thr_m['balanced_accuracy']:.4f} "
            f"thr={thr:.2f} | {elapsed:.0f}s"
        )

        if tuned_mf1 > best_val_mf1:
            best_val_mf1   = tuned_mf1
            best_threshold = thr
            patience_left  = PATIENCE
            torch.save(dict(
                model_state_dict        = model.state_dict(),
                epoch                   = epoch,
                best_val_macro_f1       = best_val_mf1,
                best_threshold          = best_threshold,
                model_name              = MODEL_NAME,
                num_labels              = 2,
                unfreeze_top_n_layers   = UNFREEZE_TOP_N_LAYERS,
            ), OUT_MODEL)
            print(f"  [✓] Saved best model → {OUT_MODEL}")
        else:
            patience_left -= 1
            print(f"  [–] Patience: {patience_left}/{PATIENCE}")
            if patience_left <= 0:
                print("[STOP] Early stopping triggered.")
                break

    # ── Final evaluation ───────────────────────────────────────
    print("\n[INFO] Loading best model for final evaluation...")
    ckpt = torch.load(OUT_MODEL, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = float(ckpt["best_threshold"])
    print(f"[INFO] Best epoch: {ckpt['epoch']}  Threshold: {best_threshold:.2f}")

    # Validation
    _, va_true, va_prob, _  = run_eval_epoch(model, val_loader,  device)
    va_pred     = (va_prob >= best_threshold).astype(np.int64)
    val_metrics = compute_metrics(va_true, va_pred)
    val_metrics["loss"]      = float(run_eval_epoch(model, val_loader, device)[0])
    val_metrics["auc_roc"]   = auc_roc(va_true, va_prob)
    val_metrics["threshold"] = best_threshold

    # Test — threshold tuning on test too (reported separately, honest)
    te_loss, te_true, te_prob, te_paths = run_eval_epoch(model, test_loader, device)
    thr_test, thr_test_m, thr_rows = tune_threshold(te_true, te_prob)

    te_pred      = (te_prob >= best_threshold).astype(np.int64)
    test_metrics = compute_metrics(te_true, te_pred)
    test_metrics["loss"]      = float(te_loss)
    test_metrics["auc_roc"]   = auc_roc(te_true, te_prob)
    test_metrics["threshold"] = best_threshold

    # Best possible test metrics (tuned on test — upper bound, reported honestly)
    te_pred_opt      = (te_prob >= thr_test).astype(np.int64)
    test_metrics_opt = compute_metrics(te_true, te_pred_opt)
    test_metrics_opt["threshold"] = thr_test
    test_metrics_opt["auc_roc"]   = auc_roc(te_true, te_prob)
    test_metrics_opt["note"]      = (
        "threshold tuned directly on test set — upper bound; "
        "use val-tuned threshold for honest reporting"
    )

    save_predictions(te_paths, te_true, te_pred, te_prob, OUT_PREDS)
    save_threshold_analysis(thr_rows, OUT_THRESH)

    final = dict(
        model               = "wav2vec2_full_attention_test",
        base_model          = MODEL_NAME,
        unfreeze_top_layers = UNFREEZE_TOP_N_LAYERS,
        train_per_class     = TRAIN_PER_CLASS,
        val_per_class       = VAL_PER_CLASS,
        test_per_class      = TEST_PER_CLASS,
        best_epoch          = int(ckpt["epoch"]),
        best_threshold_val  = best_threshold,
        best_val_macro_f1   = float(ckpt["best_val_macro_f1"]),
        val                 = val_metrics,
        test_val_threshold  = test_metrics,
        test_optimal        = test_metrics_opt,
    )
    with OUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    # ── Print report ───────────────────────────────────────────
    def pct(k, m): return f"{m[k]*100:.2f}%"

    print("\n" + "="*60)
    print("  M8 — FINAL EVALUATION REPORT")
    print("="*60)
    print(f"  Base model       : {MODEL_NAME}")
    print(f"  Unfrozen layers  : top {UNFREEZE_TOP_N_LAYERS} transformer layers")
    print(f"  Best epoch       : {ckpt['epoch']}")
    print(f"  Val threshold    : {best_threshold:.2f}")
    print()
    print("  ── VALIDATION ──────────────────────────────")
    print(f"  Accuracy         : {pct('accuracy',           val_metrics)}")
    print(f"  Balanced Accuracy: {pct('balanced_accuracy',  val_metrics)}")
    print(f"  Macro-F1         : {pct('macro_f1',           val_metrics)}")
    print(f"  AUC-ROC          : {val_metrics['auc_roc']:.4f}")
    print(f"  Recall (Stutter) : {pct('recall_stutter',     val_metrics)}")
    print(f"  Specificity      : {pct('specificity_fluent', val_metrics)}")
    print(f"  F1 Stutter       : {pct('f1_stutter',         val_metrics)}")
    print(f"  F1 Fluent        : {pct('f1_fluent',          val_metrics)}")
    print(f"  Youden J         : {val_metrics['youden_j']:.4f}")
    print(f"  TP={val_metrics['tp']} TN={val_metrics['tn']} "
          f"FP={val_metrics['fp']} FN={val_metrics['fn']}")
    print()
    print("  ── TEST (val-tuned threshold) ───────────────")
    print(f"  Accuracy         : {pct('accuracy',           test_metrics)}")
    print(f"  Balanced Accuracy: {pct('balanced_accuracy',  test_metrics)}")
    print(f"  Macro-F1         : {pct('macro_f1',           test_metrics)}")
    print(f"  AUC-ROC          : {test_metrics['auc_roc']:.4f}")
    print(f"  Recall (Stutter) : {pct('recall_stutter',     test_metrics)}")
    print(f"  Specificity      : {pct('specificity_fluent', test_metrics)}")
    print(f"  F1 Stutter       : {pct('f1_stutter',         test_metrics)}")
    print(f"  F1 Fluent        : {pct('f1_fluent',          test_metrics)}")
    print(f"  Youden J         : {test_metrics['youden_j']:.4f}")
    print(f"  TP={test_metrics['tp']} TN={test_metrics['tn']} "
          f"FP={test_metrics['fp']} FN={test_metrics['fn']}")
    print()
    print("  ── TEST (test-tuned threshold — upper bound) ─")
    print(f"  Accuracy         : {pct('accuracy',           test_metrics_opt)}")
    print(f"  Balanced Accuracy: {pct('balanced_accuracy',  test_metrics_opt)}")
    print(f"  Macro-F1         : {pct('macro_f1',           test_metrics_opt)}")
    print(f"  AUC-ROC          : {test_metrics_opt['auc_roc']:.4f}")
    print(f"  Best threshold   : {thr_test:.2f}")
    print("="*60)

    print(f"\n[OK] model    → {OUT_MODEL}")
    print(f"[OK] metrics  → {OUT_METRICS}")
    print(f"[OK] history  → {OUT_HISTORY}")
    print(f"[OK] preds    → {OUT_PREDS}")
    print(f"[OK] thresh   → {OUT_THRESH}")


if __name__ == "__main__":
    main()
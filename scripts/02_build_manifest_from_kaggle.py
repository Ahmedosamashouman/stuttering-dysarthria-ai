from pathlib import Path
import csv
import random
from collections import Counter, defaultdict

LABELS_CSV = Path("data/raw/sep28k/metadata/SEP-28k_labels.csv")
CLIPS_DIR = Path("data/raw/sep28k/clips/stuttering-clips/clips")
OUT_PATH = Path("data/processed/manifest.csv")

POSITIVE_COLUMNS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]

BAD_COLUMNS = [
    "Unsure",
    "PoorAudioQuality",
    "NoSpeech",
    "Music",
]

SEED = 42


def safe_number(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def assign_split_by_group(rows):
    groups = sorted({row["speaker_group"] for row in rows})
    random.seed(SEED)
    random.shuffle(groups)

    n = len(groups)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_groups = set(groups[:train_end])
    val_groups = set(groups[train_end:val_end])

    for row in rows:
        group = row["speaker_group"]
        if group in train_groups:
            row["split"] = "train"
        elif group in val_groups:
            row["split"] = "val"
        else:
            row["split"] = "test"

    return rows


def main():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")

    if not CLIPS_DIR.exists():
        raise FileNotFoundError(f"Missing clips folder: {CLIPS_DIR}")

    wav_files = list(CLIPS_DIR.rglob("*.wav"))
    wav_by_name = {p.name: p for p in wav_files}

    print(f"[INFO] Found wav files: {len(wav_files)}")

    rows = []

    with LABELS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        print("[INFO] Labels columns:")
        print(reader.fieldnames)

        for item in reader:
            show = str(item.get("Show", "")).strip()
            ep_id = str(item.get("EpId", "")).strip()
            clip_id = str(item.get("ClipId", "")).strip()

            filename = f"{show}_{ep_id}_{clip_id}.wav"
            path = wav_by_name.get(filename)

            if path is None:
                continue

            bad_score = 0.0
            for col in BAD_COLUMNS:
                bad_score += safe_number(item.get(col, 0))

            if bad_score > 0:
                continue

            stutter_score = 0.0
            for col in POSITIVE_COLUMNS:
                stutter_score += safe_number(item.get(col, 0))

            no_stutter_score = safe_number(item.get("NoStutteredWords", 0))

            if stutter_score > 0:
                label = "stutter"
                label_id = "1"
            elif no_stutter_score > 0:
                label = "fluent"
                label_id = "0"
            else:
                continue

            rows.append({
                "path": str(path),
                "filename": filename,
                "label": label,
                "label_id": label_id,
                "source_dataset": "SEP-28k",
                "show": show,
                "episode": ep_id,
                "clip_id": clip_id,
                "speaker_group": f"{show}_{ep_id}",
            })

    if not rows:
        raise RuntimeError("Manifest is empty. Check labels CSV and clip filenames.")

    rows = assign_split_by_group(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "filename",
        "label",
        "label_id",
        "source_dataset",
        "show",
        "episode",
        "clip_id",
        "speaker_group",
        "split",
    ]

    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)

    split_label_counts = defaultdict(Counter)
    for row in rows:
        split_label_counts[row["split"]][row["label"]] += 1

    print(f"\n[OK] Saved manifest to: {OUT_PATH}")
    print(f"\nTotal clips in manifest: {len(rows)}")

    print("\nLabel distribution:")
    for key, value in label_counts.items():
        print(f"{key}: {value}")

    print("\nSplit distribution:")
    for key, value in split_counts.items():
        print(f"{key}: {value}")

    print("\nSplit x label:")
    for split, counts in split_label_counts.items():
        print(split, dict(counts))


if __name__ == "__main__":
    main()

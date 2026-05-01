# 🧠 Speech Pathology AI — Stuttering & Dysarthria Detection

A full-stack AI system for detecting speech disorders (stuttering & dysarthria) from short audio clips using classical ML, deep learning, and self-supervised models — deployed via FastAPI and integrated into a Flutter mobile app.

---

## 🚀 Overview

This project builds an end-to-end pipeline that:

* 🎙️ Records short speech samples (3–7 seconds)
* ⚙️ Applies audio preprocessing & feature extraction
* 🧠 Runs multiple AI models (baseline → deep learning → SSL)
* 📊 Evaluates performance across paradigms
* 🌐 Deploys inference API (FastAPI)
* 📱 Connects to a real mobile app (Flutter)

---

## 🏗️ System Architecture

```
Mobile App (Flutter)
        │
        ▼
Record Audio (3–7 sec)
        │
        ▼
Preprocessing
- Resample (16kHz)
- Normalize
- Trim silence
        │
        ▼
Feature Extraction
- MFCC
- Log-Mel Spectrogram
- Spectral features
        │
        ▼
Models
- M5: GMM (baseline)
- M6: HMM (sequence)
- M7: CNN + BiLSTM
- M8: Wav2Vec2 (SSL)
        │
        ▼
FastAPI Backend
        │
        ▼
Prediction (Fluent / Stutter)
```

---

## 📊 Dataset

* Dataset size: ~16,000 audio samples
* Balanced classes:

  * 🟢 Fluent speech
  * 🔴 Stuttered speech
* Data split:

  * Training: ~70%
  * Validation: ~15%
  * Testing: ~15%

---

## ⚙️ Preprocessing Pipeline

* Convert audio to mono
* Resample to 16kHz
* Normalize amplitude
* Trim silence
* Fixed-length segmentation (3–7 seconds)

---

## 🎧 Feature Engineering

### ✅ Features Used

| Feature             | Why                                         |
| ------------------- | ------------------------------------------- |
| MFCC                | Captures speech articulation patterns       |
| Log-Mel Spectrogram | Represents frequency distribution over time |
| Spectral Features   | Captures energy, pitch, and dynamics        |

### ❌ Features Not Used

| Feature               | Reason                         |
| --------------------- | ------------------------------ |
| Raw waveform only     | Too noisy for classical models |
| High-order MFCC (>40) | Overfitting risk               |
| Formant tracking      | Complex and unstable           |

---

## 🤖 Models

### 🔹 M5 — GMM (Baseline)

* Input: MFCC vectors
* Strength: Simple, fast
* Weakness: No temporal modeling

---

### 🔹 M6 — HMM (Sequence Model)

* Input: MFCC sequences
* Strength: Captures temporal transitions
* Weakness: Limited representation power

---

### 🔹 M7 — CNN + BiLSTM (Primary Model)

* Input: Spectrograms
* CNN → spatial features
* BiLSTM → temporal dependencies
* Best balance between accuracy and efficiency

---

### 🔹 M8 — Wav2Vec2 (SSL Model)

* Pretrained: `facebook/wav2vec2-base`
* Fine-tuned for classification
* Strong feature extraction from raw audio

---

## 📈 Model Performance

| Model             | Macro F1 | Recall (Stutter) | Notes                      |
| ----------------- | -------- | ---------------- | -------------------------- |
| GMM (M5)          | 0.59     | 0.55             | Weak baseline              |
| HMM (M6)          | 0.52     | 0.79             | High recall, low precision |
| CNN + BiLSTM (M7) | 0.73     | 0.73             | Best balance               |
| Wav2Vec2 (M8)     | ~0.80+   | High             | Best overall               |

---

## 🧪 Example Output

```json
{
  "prediction": "fluent",
  "confidence": 0.90,
  "probabilities": {
    "fluent": 0.90,
    "stutter": 0.10
  },
  "model": "Wav2Vec2 Full Attention"
}
```

---

## 🌐 API (FastAPI)

### ▶️ Run locally

```bash
uvicorn api.main_deploy:app --host 0.0.0.0 --port 7860
```

### 🔍 Endpoints

* `GET /health` → check model status
* `POST /v1/predict` → upload audio file

---

## 📱 Mobile App (Flutter)

* Records audio (3 seconds default)
* Sends to API
* Displays:

  * Prediction
  * Confidence
  * Probabilities

---

## ☁️ Deployment

* Local GPU inference (CUDA enabled)
* Cloud deployment via:

  * Hugging Face Spaces (Docker)
* API accessible remotely


---

## 🧠 Key Learnings

* Importance of temporal modeling in speech
* Classical vs deep learning trade-offs
* Power of self-supervised models (Wav2Vec2)
* Real-world deployment challenges

---


* Model optimization (ONNX / TensorRT)
* Larger dataset scaling

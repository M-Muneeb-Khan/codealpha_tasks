# CodeAlpha Machine Learning Internship 🤖

**Repository**: `CodeAlpha_MachineLearning`  
**Intern**: [Your Name]  
**Domain**: Machine Learning

---

## ✅ Completed Tasks

| # | Task | Algorithm | Status |
|---|------|-----------|--------|
| 1 | Credit Scoring Model | Logistic Regression, Random Forest, XGBoost | ✅ Done |
| 2 | Emotion Recognition from Speech | CNN + BiLSTM (MFCC features) | ✅ Done |
| 3 | Handwritten Character Recognition | CNN on MNIST (99%+ accuracy) | ✅ Done |

---

## 🗂️ Project Structure

```
CodeAlpha_MachineLearning/
│
├── Task1_CreditScoring/
│   ├── credit_scoring_model.py     ← Main script
│   ├── README.md
│   └── outputs/                    ← Generated plots & CSVs
│
├── Task2_EmotionRecognition/
│   ├── emotion_recognition.py      ← Main script
│   ├── README.md
│   └── outputs/                    ← Plots & saved models
│
├── Task3_HandwrittenCharacter/
│   ├── handwritten_recognition.py  ← Main script
│   ├── README.md
│   └── outputs/                    ← Plots & saved models
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/CodeAlpha_MachineLearning.git
cd CodeAlpha_MachineLearning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run any task
python Task1_CreditScoring/credit_scoring_model.py
python Task2_EmotionRecognition/emotion_recognition.py
python Task3_HandwrittenCharacter/handwritten_recognition.py
```

> **Task 2 note**: Runs in demo mode by default (synthetic audio features).  
> To use real audio, download RAVDESS and set `data_dir` at the bottom of the script.

---

## 📊 Key Results

### Task 1 — Credit Scoring
- Best model: **XGBoost** (ROC-AUC ≈ 0.98)
- Features: debt-to-income, credit utilisation, late payments, credit score
- Class imbalance handled with **SMOTE**

### Task 2 — Emotion Recognition
- Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Features: **MFCC (40 coeff)** + Delta + Delta² + Chroma → 92 × 174
- Models: CNN + Bidirectional LSTM

### Task 3 — Handwritten Recognition
- Dataset: **MNIST** (60 000 train / 10 000 test images)
- Best accuracy: **>99%** with Deep CNN
- Data augmentation: rotation, zoom, translation

---

## 🛠️ Tech Stack
`Python 3.10+` · `TensorFlow/Keras` · `Scikit-learn` · `XGBoost` · `Librosa` · `Matplotlib` · `Seaborn`

---

*Internship at [CodeAlpha](https://www.codealpha.tech) · [LinkedIn Post](#) · [Demo Video](#)*

# Fake News Detection

This project is a **machine learning pipeline** to classify news articles as **FAKE** or **REAL** using Python and scikit-learn. The model uses TF-IDF vectorization and several classifiers to achieve high accuracy and F1-score.

---

## Features

- Load and merge Fake/Real news datasets.
- Clean and preprocess text (remove URLs, HTML, non-alphabetic characters).
- Train multiple models: Logistic Regression, Linear SVC, Multinomial Naive Bayes, Random Forest.
- Evaluate models with Stratified K-Fold Cross-Validation.
- Select the best model based on F1-score.
- Save the trained model as `.joblib`.
- Predict new news articles using the trained model.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

1. Place your datasets `Fake.csv` and `True.csv` in the project folder.
2. Run the training script:
```bash
python fake_news_detection.py
```
3. Test new news articles:
```python
from fake_news_detection import predict_news, best_pipeline

text = "Breaking news: Scientists discover cure for XYZ disease"
prediction = predict_news(best_pipeline, text)
print(prediction)  # Output: FAKE or REAL
```

---

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- joblib

---

## File Structure

```
fake-news-detection/
│
├── Fake.csv
├── True.csv
├── fake_news_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Author

Vignesh

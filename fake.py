import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
FAKE_PATH = "Fake.csv"  
TRUE_PATH = "True.csv"  
MODEL_OUTPATH = "fake_news_model.joblib"
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
def load_and_merge_datasets(fake_path: str, true_path: str) -> pd.DataFrame:
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Fake.csv or True.csv not found.")

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["label"] = "FAKE"
    df_true["label"] = "REAL"

    df_all = pd.concat([df_fake, df_true], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    if "title" in df_all.columns and "text" in df_all.columns:
        df_all["text"] = df_all["title"].astype(str).fillna("") + " " + df_all["text"].astype(str).fillna("")
    elif "text" not in df_all.columns:
        raise ValueError("Dataset must have a 'text' column or 'title'+'text' columns.")

    df_all = df_all[["text", "label"]]
    df_all = df_all.drop_duplicates(subset="text").dropna().reset_index(drop=True)
    return df_all

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text) 
    text = re.sub(r"<.*?>", " ", text)             
    text = re.sub(r"[^a-z\s]", " ", text)         
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_pipelines() -> Dict[str, Pipeline]:
    tfidf = TfidfVectorizer(
        preprocessor=basic_clean,
        stop_words="english",
        ngram_range=(1, 1),
        max_df=0.85,
        min_df=5,
        max_features=50000
    )

    return {
        "LogReg": Pipeline([
            ("tfidf", tfidf),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", tfidf),
            ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED))
        ]),
        "MultinomialNB": Pipeline([
            ("tfidf", tfidf),
            ("clf", MultinomialNB())
        ]),
        "RandomForest": Pipeline([
            ("tfidf", tfidf),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1))
        ])
    }

def evaluate_models_cv(pipelines: Dict[str, Pipeline], X, y) -> Dict[str, Dict]:
    results = {}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for name, pipe in pipelines.items():
        print(f"Evaluating {name}...")
        f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=1)
        acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=1)
        results[name] = {
            "f1_macro_mean": np.mean(f1),
            "accuracy_mean": np.mean(acc)
        }
    return results
def pick_best_model(cv_results: Dict[str, Dict]) -> str:
    return max(cv_results, key=lambda name: (cv_results[name]["f1_macro_mean"], cv_results[name]["accuracy_mean"]))

def final_train_and_eval(pipeline: Pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n=== Final Test Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return pipeline
def predict_news(model_pipeline: Pipeline, text: str) -> str:
    text_clean = basic_clean(text)
    prediction = model_pipeline.predict([text_clean])
    return prediction[0]
if __name__ == "__main__":
    df = load_and_merge_datasets(FAKE_PATH, TRUE_PATH)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

    pipelines = build_pipelines()
    cv_results = evaluate_models_cv(pipelines, X_train, y_train)

    print("\nCross-validation Results:")
    for name, res in cv_results.items():
        print(f"{name}: F1={res['f1_macro_mean']:.4f}, Accuracy={res['accuracy_mean']:.4f}")

    best_name = pick_best_model(cv_results)
    print(f"\nBest Model Selected: {best_name}")

    best_pipeline = pipelines[best_name]
    best_pipeline = final_train_and_eval(best_pipeline, X_train, y_train, X_test, y_test)

    joblib.dump(best_pipeline, MODEL_OUTPATH)
    print(f"\nModel saved to {MODEL_OUTPATH}")

    sample_text = "Breaking news: Scientists discover cure for XYZ disease"
    print(f"\nPrediction for sample text:\n'{sample_text}' -> {predict_news(best_pipeline, sample_text)}")

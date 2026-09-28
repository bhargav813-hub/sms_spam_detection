"""
SMS Spam Detection System - Model Training & Evaluation Pipeline
Loads dataset, applies NLP preprocessing, extracts TF-IDF features,
trains and compares Multinomial Naive Bayes and Logistic Regression,
evaluates performance dynamically, selects the winning model based on F1-score,
and saves the trained artifacts using joblib.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_text


def load_and_clean_data(filepath: str = "data/spam.csv") -> tuple[pd.DataFrame, dict]:
    """
    Loads raw SMS spam dataset, drops redundant columns, cleans missing values,
    and removes duplicates.
    
    Returns:
        tuple[pd.DataFrame, dict]: Cleaned dataframe and EDA metadata dictionary.
    """
    # Defensive path check
    if not os.path.exists(filepath):
        alt_path = "spam.csv"
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. Please verify the dataset file exists."
            )

    print(f"[1/6] Loading raw dataset from: {filepath}")
    # Read with latin-1 encoding as standard for the SMS Spam collection
    try:
        df_raw = pd.read_csv(filepath, encoding="latin-1")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(filepath, encoding="utf-8", errors="replace")

    initial_rows, initial_cols = df_raw.shape
    print(f"      Initial shape: {initial_rows} rows, {initial_cols} columns")

    # Keep only relevant columns (v1 -> label, v2 -> message)
    if "v1" in df_raw.columns and "v2" in df_raw.columns:
        df = df_raw[["v1", "v2"]].copy()
        df.rename(columns={"v1": "label", "v2": "message"}, inplace=True)
    elif "label" in df_raw.columns and "message" in df_raw.columns:
        df = df_raw[["label", "message"]].copy()
    else:
        raise ValueError(
            f"Dataset does not contain expected columns ('v1', 'v2') or ('label', 'message'). Found: {list(df_raw.columns)}"
        )

    # Missing values analysis
    missing_counts = df.isnull().sum().to_dict()
    df.dropna(subset=["label", "message"], inplace=True)

    # Standardize label values
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    valid_labels = {"ham", "spam"}
    df = df[df["label"].isin(valid_labels)].copy()

    # Duplicate detection and removal
    duplicate_count = int(df.duplicated(subset=["message"]).sum())
    print(f"      Identified {duplicate_count} duplicate messages. Removing duplicates...")
    df.drop_duplicates(subset=["message"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    final_rows = len(df)
    print(f"      Final cleaned dataset: {final_rows} unique messages")

    # Character and word length statistics before NLP preprocessing
    df["char_count"] = df["message"].apply(len)
    df["word_count"] = df["message"].apply(lambda msg: len(msg.split()))

    ham_subset = df[df["label"] == "ham"]
    spam_subset = df[df["label"] == "spam"]

    eda_stats = {
        "initial_rows": initial_rows,
        "initial_cols": initial_cols,
        "cleaned_rows": final_rows,
        "duplicates_removed": duplicate_count,
        "missing_values": missing_counts,
        "class_distribution": df["label"].value_counts().to_dict(),
        "class_percentages": (df["label"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "ham_count": len(ham_subset),
        "spam_count": len(spam_subset),
        "ham_char_mean": float(ham_subset["char_count"].mean()),
        "spam_char_mean": float(spam_subset["char_count"].mean()),
        "ham_word_mean": float(ham_subset["word_count"].mean()),
        "spam_word_mean": float(spam_subset["word_count"].mean()),
    }

    return df, eda_stats


def extract_top_words(df: pd.DataFrame, top_n: int = 15) -> dict:
    """
    Computes top N most frequent words in preprocessed spam and ham messages.
    """
    ham_words = []
    spam_words = []

    for _, row in df.iterrows():
        tokens = row["cleaned_message"].split()
        if row["label"] == "ham":
            ham_words.extend(tokens)
        else:
            spam_words.extend(tokens)

    return {
        "top_ham_words": Counter(ham_words).most_common(top_n),
        "top_spam_words": Counter(spam_words).most_common(top_n),
    }


def train_and_evaluate(
    data_path: str = "data/spam.csv",
    output_dir: str = "models",
    random_state: int = 42
) -> dict:
    """
    Executes the entire NLP ML pipeline:
    1. Ingestion & Data Cleaning
    2. NLP Preprocessing (lowercase, punct/symbols, tokenize, stopwords, stem)
    3. 80/20 Stratified Train/Test Split
    4. TF-IDF Feature Extraction (no data leakage)
    5. Training MultinomialNB & LogisticRegression
    6. Performance Evaluation & Dynamic Best Model Selection
    7. Saving artifacts with joblib
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and clean data
    df, eda_stats = load_and_clean_data(data_path)

    # 2. NLP Preprocessing
    print("[2/6] Applying NLP preprocessing (cleaning, tokenization, stopword removal, stemming)...")
    df["cleaned_message"] = df["message"].apply(preprocess_text)

    # Compute top words for EDA
    top_words = extract_top_words(df, top_n=15)
    eda_stats.update(top_words)

    # Encode labels (ham -> 0, spam -> 1)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["cleaned_message"]
    y = df["target"]

    # 3. Stratified 80/20 Train-Test Split
    print("[3/6] Performing Stratified 80/20 Train-Test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    print(f"      Training set: {len(X_train)} samples")
    print(f"      Testing set:  {len(X_test)} samples")

    # 4. TF-IDF Feature Extraction (Prevent Data Leakage)
    print("[4/6] Fitting TfidfVectorizer on training set and transforming test set...")
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"      TF-IDF Vocabulary size: {X_train_tfidf.shape[1]} features")

    # 5. Model Training & Comparison
    print("[5/6] Training and evaluating Machine Learning models...")
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        )
    }

    results = {}
    comparison_records = []

    for name, model in models.items():
        print(f"      --> Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }

        comparison_records.append({
            "Model": name,
            "Accuracy": round(acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1-Score": round(f1 * 100, 2)
        })

    comparison_df = pd.DataFrame(comparison_records)
    print("\n--- Model Comparison Table ---")
    print(comparison_df.to_string(index=False))

    # 6. Dynamic Best Model Selection based on F1-score
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model_obj = results[best_model_name]["model"]
    best_metrics = results[best_model_name]

    print(f"\n[6/6] Winning Model Selected: {best_model_name}")
    print(f"      F1-Score: {best_metrics['f1_score']:.4f} ({best_metrics['f1_score']*100:.2f}%)")
    print(f"      Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f"      Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
    print(f"      Recall: {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")

    # Serialize artifacts
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    classifier_path = os.path.join(output_dir, "spam_classifier.pkl")
    metrics_path = os.path.join(output_dir, "model_metrics.pkl")
    nb_path = os.path.join(output_dir, "naive_bayes.pkl")
    lr_path = os.path.join(output_dir, "logistic_regression.pkl")

    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(best_model_obj, classifier_path)
    joblib.dump(models["Multinomial Naive Bayes"], nb_path)
    joblib.dump(models["Logistic Regression"], lr_path)

    # Save complete evaluation bundle (without model objects to keep metrics lightweight and pickle-safe)
    eval_bundle = {
        "best_model_name": best_model_name,
        "best_metrics": {
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1_score"],
            "confusion_matrix": best_metrics["confusion_matrix"],
            "classification_report": best_metrics["classification_report"]
        },
        "all_models_metrics": {
            k: {
                "accuracy": v["accuracy"],
                "precision": v["precision"],
                "recall": v["recall"],
                "f1_score": v["f1_score"],
                "confusion_matrix": v["confusion_matrix"],
            } for k, v in results.items()
        },
        "comparison_table": comparison_df.to_dict(orient="records"),
        "eda_stats": eda_stats,
        "test_size": 0.20,
        "random_state": random_state,
        "feature_count": X_train_tfidf.shape[1]
    }

    joblib.dump(eval_bundle, metrics_path)
    print(f"\nSaved artifacts successfully:")
    print(f"  - {vectorizer_path}")
    print(f"  - {classifier_path}")
    print(f"  - {metrics_path}")

    return eval_bundle


if __name__ == "__main__":
    data_file = "data/spam.csv"
    train_and_evaluate(data_path=data_file)

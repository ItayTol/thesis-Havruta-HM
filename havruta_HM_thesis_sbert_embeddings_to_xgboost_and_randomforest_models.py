# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 20:41:49 2025

@author: User
"""

#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss, matthews_corrcoef, cohen_kappa_score,
    top_k_accuracy_score, roc_auc_score)
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

def load_posts_and_trait_true_label(personality_trait: str):
    df = pd.read_excel("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/true_labels_3_classes.xlsx")
    # for _, row in df.iterrows():
    #       row['post1'], row['post2'] = row['post1'].replace("â€™","'"), row['post2'].replace("â€™","'")
    return df[['p', 'post1', 'post2', f'{personality_trait}']]


def main(df, post1="post1", post2="post2", target="e", sbert_model="sentence-transformers/all-MiniLM-L6-v2"):
    # make text
    df["text"] = df[post1].astype(str).fillna("") + " " + df[post2].astype(str).fillna("")
    # pick target (fallbacks)
    df = df[["text", target]].dropna().reset_index(drop=True).rename(columns={target: "label"})
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    sbert = SentenceTransformer(sbert_model)
    X = sbert.encode(df["text"].tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True)

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    rows = []
    for i, (tr, te) in enumerate(skf.split(X, y), start=1):
        xtr, xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        clf = XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
            num_class=len(le.classes_), reg_lambda=1.0, random_state=42,
            tree_method="auto", n_jobs=-1
        )
        clf.fit(xtr, ytr)
        proba = clf.predict_proba(xte)
        pred = proba.argmax(axis=1)
        row = {
            "fold": i,
            "accuracy": accuracy_score(yte, pred),
            "precision_macro": precision_score(yte, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(yte, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(yte, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(yte, pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
            "matthews_corrcoef": matthews_corrcoef(yte, pred),
            "cohen_kappa": cohen_kappa_score(yte, pred),
            "log_loss": log_loss(yte, proba, labels=np.arange(len(le.classes_))),
            "top_2_accuracy": top_k_accuracy_score(yte, proba, k=2, labels=np.arange(len(le.classes_))) if len(le.classes_) >= 2 else np.nan
        }
        try:
            row["roc_auc_ovr_macro"] = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
            row["roc_auc_ovr_weighted"] = roc_auc_score(yte, proba, multi_class="ovr", average="weighted")
        except Exception:
            row["roc_auc_ovr_macro"] = np.nan
            row["roc_auc_ovr_weighted"] = np.nan
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.loc["mean"] = metrics.drop(columns=["fold"]).mean(numeric_only=True)
    print(metrics)

if __name__ == "__main__":
    # Usage: python sbert_xgb_2fold.py /path/to/true_labels_3_classes.csv
    import sys
    # assert len(sys.argv) >= 2, "Please pass the CSV path as the first argument."
    main(load_posts_and_trait_true_label('e'))

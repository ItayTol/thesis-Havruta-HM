# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 20:41:49 2025

@author: User
"""


from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
    """
    Plots and optionally saves a confusion matrix with white-colored axis ticks.

    Parameters:
    - y_true: list of true class labels
    - y_pred: list of predicted class labels
    - labels: list of class labels (e.g., ["Low", "Moderate", "High"])
    - title: plot title
    - save_path: file path to save the image (e.g., "cmatrix.png"). If None, does not save.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted", color='white')
    plt.ylabel("True", color='white')
    plt.title(title, color='white')
    plt.xticks(color='red')
    plt.yticks(color='red')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor='black', dpi=300)
    
    plt.show()


personality_trait = 'e'
def load_posts_and_trait_true_label(personality_trait: str):
    df = pd.read_csv("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/true_labels_3_classes.csv", encoding='cp1252')
    return df[['p', 'post1', 'post2', f'{personality_trait}']]

df = load_posts_and_trait_true_label(personality_trait)

df['text'] = df['post1'] + df['post2']
df['label'] = df[personality_trait]
model = SentenceTransformer('all-mpnet-base-v2')

# Function to generate embeddings
def generate_embeddings(texts):
  return model.encode(texts)

rf_accuracies = []
xgb_accuracies = []

for _ in range(5):
    # Split data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df = df[:50]
    test_df = df[50:60] # Last 10 samples

    # Generate embeddings
    X_train = generate_embeddings(train_df['text'].tolist())
    X_test = generate_embeddings(test_df['text'].tolist())
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_accuracies.append(rf_accuracy)
    all_rf_pred.append(rf_predictions)
    # XGBoost
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_accuracies.append(xgb_accuracy)
    
plot_confusion_matrix(y_true, y_pred, labels, title = 'rf_pred')


# Report results
print("Random Forest:")
print(f"Mean Accuracy: {np.mean(rf_accuracies):.4f}")
print(f"Standard Deviation: {np.std(rf_accuracies):.4f}")

print("\nXGBoost:")
print(f"Mean Accuracy: {np.mean(xgb_accuracies):.4f}")
print(f"Standard Deviation: {np.std(xgb_accuracies):.4f}")

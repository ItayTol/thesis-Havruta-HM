# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 20:20:53 2025

@author: User
"""

import openai
import numpy as np
# import re
from plyer import notification
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime
# from havruta_HM_thesis_prompts_3_classes import load_trait
from havruta_HM_thesis_load_rules_3_classes import load_rules_with_gpt
from havruta_HM_thesis_classification import classifier_zero_and_few_shot
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from  havruta_HM_hyperparams_grid import VARIANTS, evaluate_variant_on_df
from havruta_HM_thesis_classification_prep import evaluate_zero_shot, evaluate_few_shot


openai.api_key = os.getenv('OPENAI_API_KEY')
pd.set_option('display.max_colwidth', None)  # Show full text in cells

    # ---- small helpers ---------------------------------------------------------

def load_trait(personality_trait: str):
    traits = {'o':'openness to experiences',
              'c': 'conscientiousness',
              'e': 'extraversion',
              'a': 'aggreeableness',
              'n': 'neuroticism'}
    return traits[personality_trait]

# Load dataset (social media posts + true labels)
def load_posts_and_trait_true_label(personality_trait: str):
    df = pd.read_excel("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/true_labels_3_classes.xlsx")
    return df[['p', 'post1', 'post2', f'{personality_trait}']][0:20]

def time_stamp(text: str):
        now = datetime.now()
        now_txt = now.strftime("%Y-%m-%d_%H-%M-%S")
        print(text, now_txt)
        return now_txt
  
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

def build_few_shot_prompt(train_df: pd.DataFrame, personality_trait: str, k: int = 5, seed: int = 0) -> str:
    """Builds a short few-shot block from the training slice (UTF-8 safe)."""
    rng = np.random.RandomState(seed)
    k = min(k, len(train_df))
    if k == 0:
        return ""
    rows = train_df.sample(k, random_state=rng)
    lines = []
    trait_name = load_trait(personality_trait)
    for i, row in rows.iterrows():
        lines.append(
            f"Example {i}\n"
            f"Texts:\n1. {row['post1']}\n2. {row['post2']}\n"
            f"{trait_name} level: {row[personality_trait]}\n"
        )
    return "\n".join(lines)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def _compute_metric(df: pd.DataFrame) -> float:
    """Mean accuracy based on 'ok' or 'score_rules'."""
    if "ok" in df.columns:
        return float(np.mean(df["ok"]))
    if "score_rules" in df.columns:
        return float(np.mean(df["score_rules"]))
    raise ValueError("No usable metric column found.")

def _json_dump_safely(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    
# ---- main pipeline ---------------------------------------------------------
"""
Unified pipeline for rule-based, zero-shot, and few-shot evaluation
Outer: n shuffle splits
Inner: k-fold CV on DEV to select best variant
"""


# -------------------------------
# Main pipeline
# -------------------------------

def run_pipeline(
    personality_trait: str,
    outer_reps: int = 1,
    test_n: int = 5,
    k_inner_folds: int = 4,
    seed: int = 42,
    results_dir_root: str = r"C:/Users/User/OneDrive/Desktop/Masters/thesis/Results/"
):
    participants = load_posts_and_trait_true_label(personality_trait)

    # --- results folders ---
    trait_folder = f"{load_trait(personality_trait)}/"
    base_path = os.path.join(
        results_dir_root,
        trait_folder,
        f"run_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(base_path, exist_ok=True)

    subdirs = {
        "val": os.path.join(base_path, "validation_set_results"),
        "test": os.path.join(base_path, "test_set_results"),
        "meta": os.path.join(base_path, "_meta"),
        "rules": os.path.join(base_path, "_rules")  # central place if you want a flat copy too
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)

    outer_results = []

    X, y = participants[['p', 'post1', 'post2']], participants[personality_trait]
    for outer_rep in range(1, outer_reps + 1):
        # ---- Outer split ----
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, stratify = y, 
                                                        test_size = test_n, random_state=42)
        # Persist indices (provenance)
        idx_dir = os.path.join(subdirs["meta"], f"OUTER{outer_rep}")
        os.makedirs(idx_dir, exist_ok=True)
        pd.DataFrame({"idx": X_dev['p']}).to_csv(os.path.join(idx_dir, "DEV_IDX.csv"), index=False, encoding="utf-8")
        pd.DataFrame({"idx": X_test['p']}).to_csv(os.path.join(idx_dir, "TEST_IDX.csv"), index=False, encoding="utf-8")

        # ---- Load rules (tied to DEV participants if your loader supports it) ----
        try:
            rules_all = load_rules_with_gpt(personality_trait, X_dev['p'].tolist())
        except TypeError as e:
            return e

        # Save the exact rules used this rep (one copy in meta and also per-VAL/TEST locations below)
        rules_rep_dir = os.path.join(subdirs["rules"], f"OUTER{outer_rep}")
        os.makedirs(rules_rep_dir, exist_ok=True)
        _json_dump_safely(rules_all, os.path.join(rules_rep_dir, f"RULES-OUTER{outer_rep}.json"))


        # ---- Inner CV: select best variant ----
        skf = StratifiedKFold(n_splits = k_inner_folds)
        for fold_i, (train_index, test_index) in enumerate(skf.split(X_dev, y_dev), 1):
            x_inner_train_fold, x_inner_test_fold = X_dev.iloc[train_index, :], X_dev.iloc[test_index, :] # (40 - k_inner_folds, 3) , (40 - k_inner_folds, 1)
            y_inner_train_fold, y_inner_test_fold = y_dev.iloc[train_index], y_dev.iloc[test_index] # (k_inner_folds, 3) , (k_inner_folds, 1)

            inner_train_fold = pd.DataFrame(pd.concat([x_inner_train_fold, y_inner_train_fold], axis=1))
            inner_test_fold = pd.DataFrame(pd.concat([x_inner_test_fold, y_inner_test_fold], axis=1))

            val_metrics = []
            for v in VARIANTS:
                vid = v["variant_id"]
                print("    Variant" ,vid, "starting")
                fold_scores = []
                print("        Fold:", fold_i, "  ->   Inner Train:", len(x_inner_train_fold), " ; Inner Val:", len(x_inner_test_fold))
                
                
                # Folder per variant for neatness + accuracy per variant snapshot
                variant_fold_dir = os.path.join(subdirs["val"], f"OUTER{outer_rep}", f"{vid}", f"Fold_{fold_i}")
                os.makedirs(variant_fold_dir, exist_ok=True)
                
                
                # ---- Load rules (tied to inner train participants) ----
                try:
                    inner_rules = load_rules_with_gpt(personality_trait, inner_train_fold['p'].tolist())
                except TypeError as e:
                    return e
                
                  # Export rules used for this validation fold (same object, but snapshot it here)
                _json_dump_safely(inner_rules, os.path.join(variant_fold_dir, "RULES_USED.json"))
               
                # Evaluate
                df_val = evaluate_variant_on_df(v, personality_trait, inner_test_fold, inner_rules)
                score = _compute_metric(df_val)
                fold_scores.append(score)

                # Save per-fold val results
                df_val.to_csv(
                    os.path.join(variant_fold_dir, f"VAL-OUTER{outer_rep}-VAR{vid}-FOLD{fold_i}.csv"),
                    index=False, encoding="utf-8"
                )

            val_metrics.append((vid, np.mean(fold_scores) if fold_scores else float("nan")))

        best_vid, best_score = max(val_metrics, key=lambda x: (x[1], str(x[0])))
        best_variant = next(v for v in VARIANTS if v["variant_id"] == best_vid)
        print(f"[OUTER {outer_rep}] Best variant={best_vid} (VAL score={best_score:.3f})")

        # ---- Build few-shot prompt examples from DEV ----
        dev = pd.DataFrame(pd.concat([X_dev, y_dev], axis=1))
        test = pd.DataFrame(pd.concat([X_test, y_test], axis=1))
        
        few_shot_examples = build_few_shot_prompt(dev , personality_trait)
        few_dir = os.path.join(subdirs["meta"], f"OUTER{outer_rep}")
        os.makedirs(few_dir, exist_ok=True)
        
        # Store few-shot block (helps tracing)
        with open(os.path.join(few_dir, "FEW_SHOT_EXAMPLES.txt"), "w", encoding="utf-8") as f:
            if isinstance(few_shot_examples, (dict, list)):
                f.write(json.dumps(few_shot_examples, ensure_ascii=False, indent=2))
            else:
                f.write(str(few_shot_examples))

        # ---- TEST eval ----
        summary_results_test = []

        test_rep_dir = os.path.join(subdirs["test"], f"OUTER{outer_rep}")
        os.makedirs(test_rep_dir, exist_ok=True)

        # Export the same rules for TEST (snapshot + hash)
        _json_dump_safely(rules_all, os.path.join(test_rep_dir, "RULES_USED.json"))

        # (a) Rule-based best variant
        # best_v = next(v for v in VARIANTS if v["variant_id"] == "vA")
        # best_vid = "vA"

        df_rule = evaluate_variant_on_df(best_variant, personality_trait, test, rules_all)
        acc_rule = _compute_metric(df_rule)
        df_rule.to_csv(os.path.join(test_rep_dir, f"TEST-OUTER{outer_rep}-VAR{best_vid}.csv"), index=False, encoding="utf-8")
        summary_results_test.append({"outer_rep": outer_rep, "method": "rules", "variant": best_vid, "acc": acc_rule})

        # (b) Zero-shot
        preds_zero = []
        reses_zero = []
        expla_zero = []
        for i, row in test.iterrows():
            res = evaluate_zero_shot(personality_trait, row["post1"], row["post2"])
            preds_zero.append(res["level"])
            expla_zero.append(res['explanation'])
            reses_zero.append(res)

        # try:
        #     pass
        # except Exception as e:
        #     print("Something is wrong with evaluate_zero_shot", e)
        acc_zero = np.mean([pred == true for pred, true in zip(preds_zero, test[personality_trait])])
        df_res_zero = pd.DataFrame({"p": test.get("p"), "pred_zero": preds_zero, 'explanation_zero': expla_zero, "res_zero": reses_zero})
        df_res_zero.to_csv(os.path.join(test_rep_dir, "TEST-OUTER{0}-ZERO.csv".format(outer_rep)), index=False, encoding="utf-8")
        summary_results_test.append({"outer_rep": outer_rep, "method": "zero_shot", "variant": "zero", "acc": float(acc_zero)})

        # (c) Few-shot (using DEV-derived examples)
        preds_few = []
        reses_few = []
        expla_few = []
        for _, row in test.iterrows():
            res = evaluate_few_shot(personality_trait, row["post1"], row["post2"], few_shot=few_shot_examples)
            reses_few.append(res)
            preds_few.append(res['level'])
            expla_few.append(res['explanation'])
        acc_few = np.mean([pred == true for pred, true in zip(preds_few, test[personality_trait])])
        df_res_few = pd.DataFrame({"p": test.get("p", pd.Series(range(len(test)))), "pred_few": preds_few, 'explanation_few':expla_few, "res_few": reses_few})
        df_res_few.to_csv(
            os.path.join(test_rep_dir, "TEST-OUTER{0}-FEW.csv".format(outer_rep)), index=False, encoding="utf-8"
        )
        summary_results_test.append({"outer_rep": outer_rep, "method": "few_shot", "variant": "few", "acc": float(acc_few)})

        # Save outer summary
        pd.DataFrame(summary_results_test).to_csv(os.path.join(test_rep_dir, f"TEST-OUTER{outer_rep}-SUMMARY.csv"),
                                          index=False, encoding="utf-8")
        
        rule_zero_merged = pd.merge(df_rule, df_res_zero, on = "p", how = "inner")
        test_full_results = pd.merge(rule_zero_merged, df_res_few, on = "p", how = "inner")
        test_full_results.to_csv(
            os.path.join(test_rep_dir, f"TEST-OUTER{outer_rep}-FULL-RESULTS.csv"), index=False, encoding="utf-8"
        )
        
        outer_results.extend(summary_results_test)

    # --- Final summary ---
    df_summary = pd.DataFrame(outer_results)
    df_summary.to_csv(os.path.join(subdirs["meta"], "OUTER_SUMMARY.csv"), index=False, encoding="utf-8")
    print("\nFinal results:")
    print(df_summary.groupby("method")["acc"].agg(["mean", "std"]))

    return df_summary


if __name__ == "__main__":

    
    try:
        start = datetime.now()
        run_pipeline('o')

        print(f'\nTime elapsed: {datetime.now() - start}')

        # Notify on success
        notification.notify(
            title="✅ Code Completed",
            message="The script finished successfully.",
            timeout=5)
    except Exception as e:
        # Notify on error
        error_message = f"🚨 ERROR: {type(e).__name__} - {str(e)}"
        notification.notify(
            title="❌ Code Crashed",
            message=error_message,
            timeout=10
        )
        print(f'\nTime elapsed: {datetime.now() - start}')

        # Optional: print full traceback to console or file
        traceback.print_exc()
        # You can also log it to a file if needed:
        with open("C:/Users/User/OneDrive/Desktop/Masters/thesis/code 3 classes/error_log.txt", "w", encoding='utf-8') as f:
            f.write(traceback.format_exc())  # main_recursive()
    end = datetime.now()
    end_txt = end.strftime("%Y-%m-%d_%H-%M-%S")
    print('END', end_txt)
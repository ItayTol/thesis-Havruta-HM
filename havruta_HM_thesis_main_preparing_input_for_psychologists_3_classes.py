# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 12:20:40 2025

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
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
# from havruta_HM_thesis_prompts_3_classes import load_trait
from havruta_HM_thesis_load_rules_3_classes import load_rules_with_gpt
from havruta_HM_thesis_classification import rank_rules_and_classify
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
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
    df = pd.read_excel("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/105_participants_data.xlsx", sheet_name="summary_true_labels_3_classes")
    return df[['p', 'post1', 'post2', f'{personality_trait}']]

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

def build_few_shot_prompt(train_df: pd.DataFrame, personality_trait: str, k: int = 5) -> str:
    """Builds a short few-shot block from the training slice (UTF-8 safe)."""
    rows = train_df.sample(k)
    lines = []
    trait_name = load_trait(personality_trait)
    for i, row in rows.iterrows():
        lines.append(
            f"Example {i+1}\n"
            f"Texts:\n1. {row['post1']}\n2. {row['post2']}\n"
            f"{trait_name} level: {row[personality_trait]}\n"
        )
    return "\n".join(lines)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)
    
def _json_dump_safely(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    
# ---- main pipeline ---------------------------------------------------------
"""
Unified pipeline for rule-based, zero-shot, and few-shot evaluation
Outer: n shuffle splits
"""


# -------------------------------
# Main pipeline
# -------------------------------

def run_pipeline(
    personality_trait: str,
    outer_reps: int,
    # test_n: int = 30,
    seed: int = 42,
    results_dir_root: str = r"C:/Users/User/OneDrive/Desktop/Masters/thesis/Results/"
):
    participants = load_posts_and_trait_true_label(personality_trait)
    used_for_rules_40_participants = participants[:40]
    
    # --- results folders ---
    trait_folder = f"{load_trait(personality_trait)}/"
    base_path = os.path.join(
        results_dir_root,
        trait_folder,
        f"run_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(base_path, exist_ok=True)

    subdirs = {
        # "val": os.path.join(base_path, "validation_set_results"),
        "test": os.path.join(base_path, "test_set_results"),
        "meta": os.path.join(base_path, "_meta"),
        "rules": os.path.join(base_path, "_rules")  # central place if you want a flat copy too
    }
    for p in subdirs.values():
        os.makedirs(p, exist_ok=True)

    summary_results_test = []
    results_test_list = []
    
    X, y = used_for_rules_40_participants[['p', 'post1', 'post2']], used_for_rules_40_participants[personality_trait]
    for outer_rep in range(1, outer_reps + 1):
        results_rep_test_list = []
        # ---- Outer split ----
        # X_dev, X_test, y_dev, y_test = train_test_split(X, y, stratify = y, test_size = test_n, random_state=42)
        skf = StratifiedKFold(n_splits=2, shuffle=True)        
        for fold_index, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            x_train_fold, x_test_fold = X.loc[train_index], X.loc[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
             # Persist indices (provenance)

            idx_dir = os.path.join(subdirs["meta"], f"OUTER{outer_rep}")
            os.makedirs(idx_dir, exist_ok = True)
            pd.DataFrame({"idx": sorted(x_train_fold['p'])}).to_csv(os.path.join(idx_dir, f"TRAIN_IDX_fold{fold_index}.csv"), index=False, encoding="utf-8")
            pd.DataFrame({"idx": sorted(x_test_fold['p'])}).to_csv(os.path.join(idx_dir, f"TEST_IDX_fold{fold_index}.csv"), index=False, encoding="utf-8")
    
            # ---- Load rules (tied to DEV participants if your loader supports it) ----
            try:
                rules = load_rules_with_gpt(personality_trait, x_train_fold['p'].tolist())
            except TypeError as e:
                return e
    
            # Save the exact rules used this rep (one copy in meta and also per-VAL/TEST locations below)
            rules_rep_dir = os.path.join(subdirs["rules"], f"OUTER{outer_rep}")
            os.makedirs(rules_rep_dir, exist_ok=True)
            _json_dump_safely(rules, os.path.join(rules_rep_dir, f"RULES-OUTER{outer_rep}_fold{fold_index}.json"))
      
            # ---- Build few-shot prompt examples from DEV ----
            dev = pd.DataFrame(pd.concat([x_train_fold, y_train_fold], axis=1))
            test = pd.DataFrame(pd.concat([x_test_fold, y_test_fold], axis=1))
            
            few_shot_examples = build_few_shot_prompt(dev, personality_trait)
            few_dir = os.path.join(subdirs["meta"], f"OUTER{outer_rep}")
            os.makedirs(few_dir, exist_ok=True)
            
            # Store few-shot block (helps tracing)
            with open(os.path.join(few_dir, f"FEW_SHOT_EXAMPLES_fold{fold_index}.txt"), "w", encoding="utf-8") as f:
                if isinstance(few_shot_examples, (dict, list)):
                    f.write(json.dumps(few_shot_examples, ensure_ascii=False, indent=2))
                else:
                    f.write(str(few_shot_examples))
    
            # ---- TEST eval ----
            test_rep_dir = os.path.join(subdirs["test"])    
            
            # (a) Rule-based best variant
            preds_rule = []
            reses_rule = []
            expla_rule = []
            ok_rule = []
            for _, row in test.iterrows():
                res = rank_rules_and_classify(personality_trait, row["post1"], row["post2"], rules)
                indented_res = json.dumps(res, ensure_ascii=False, indent=2)
                reses_rule.append(indented_res)
                preds_rule.append(res["trait_classification"]["classified_level"])
                expla_rule.append(res["trait_classification"]["justification"])
            ok_rule = [int(pred == true) for pred, true in zip(preds_rule, test[personality_trait])]
            acc_rule = np.mean([pred == true for pred, true in zip(preds_rule, test[personality_trait])])
            df_res_rule = pd.DataFrame({'rep': outer_rep, "p": test.get("p"), "post1": test["post1"], "post2": test["post2"],
                                        'true': test[personality_trait], "pred_rules": preds_rule, 'explanation_rules':expla_rule,
                                        'res_rules': reses_rule, 'accuracy_rules': ok_rule})
           
            # df_res_rule.to_csv(os.path.join(test_rep_dir, f"TEST-OUTER{outer_rep}.csv"), index=False, encoding="utf-8")
            summary_results_test.append({"outer_rep": outer_rep, 
                                        "fold": fold_index,
                                        "method": "rules_based", 
                                        "acc": float(acc_rule)})
            # (b) Zero-shot
            preds_zero = []
            reses_zero = []
            expla_zero = []
            ok_zero = []
            for i, row in test.iterrows():
                res = evaluate_zero_shot(personality_trait, row["post1"], row["post2"])
                indented_res = json.dumps(res, ensure_ascii=False, indent=2)
                reses_zero.append(indented_res)
                preds_zero.append(res["level"])
                expla_zero.append(res['explanation'])
            acc_zero = np.mean([pred == true for pred, true in zip(preds_zero, test[personality_trait])])
            ok_zero = [int(pred == true) for pred, true in zip(preds_zero, test[personality_trait])]
            df_res_zero = pd.DataFrame({"p": test.get("p"), "pred_zero": preds_zero, 'explanation_zero': expla_zero, "accuracy_zero": ok_zero})
            # df_res_zero.to_csv(os.path.join(test_rep_dir, "TEST-OUTER{0}-ZERO.csv".format(outer_rep)), index=False, encoding="utf-8")
            summary_results_test.append({"outer_rep": outer_rep, 
                                         "fold": fold_index,
                                         "method": "zero_shot", 
                                         "acc": float(acc_zero)})
    
            # (c) Few-shot (using DEV-derived examples)
            preds_few = []
            reses_few = []
            expla_few = []
            ok_few = []
           
            for _, row in test.iterrows():
                res = evaluate_few_shot(personality_trait, row["post1"], row["post2"], few_shot=few_shot_examples)
                indented_res = json.dumps(res, ensure_ascii=False, indent=2)
                reses_few.append(indented_res)
                preds_few.append(res['level'])
                expla_few.append(res['explanation'])
            acc_few = np.mean([pred == true for pred, true in zip(preds_few, test[personality_trait])])
            ok_few = [int(pred == true) for pred, true in zip(preds_few, test[personality_trait])]
            df_res_few = pd.DataFrame({"p": test.get("p"), "pred_few": preds_few, 'explanation_few':expla_few, "accuracy_few": ok_few})
            # df_res_few.to_csv(os.path.join(test_rep_dir, "TEST-OUTER{0}-FEW.csv".format(outer_rep)), index=False, encoding="utf-8")
            summary_results_test.append({"outer_rep": outer_rep, 
                                         "fold": fold_index,
                                         "method": "few_shot", 
                                         "acc": float(acc_few)})
            
            rule_zero_merged = pd.merge(df_res_rule, df_res_zero, on = "p", how = "inner")
            results_fold_test = pd.merge(rule_zero_merged, df_res_few, on = "p", how = "inner")
            results_rep_test_list.append(results_fold_test)
            
        # Save outer rep summary
        results_rep_test = pd.concat(results_rep_test_list)
        results_rep_test.to_csv(os.path.join(test_rep_dir, f"TEST-FULL-Rep{outer_rep}-RESULTS.csv"), index=False, encoding="utf-8")

        results_test_list.append(results_rep_test)

        df_rep_summary = pd.DataFrame(summary_results_test)
        df_rep_summary.to_csv(os.path.join(test_rep_dir, f"TEST-Rep{outer_rep}-SUMMARY.csv"),
                                          index=False, encoding="utf-8")
        
        
        print(f"\n{outer_rep}/{outer_reps} Rep results:")

        print(df_rep_summary.groupby("method")["acc"].agg(["mean", "std"]))
        


    # --- Final summary ---
    results_test = pd.concat(results_test_list) 
    results_test.to_csv(os.path.join(test_rep_dir, "TEST-FULL-RESULTS.csv"), index=False, encoding="utf-8")
    try:
        true = results_test['true']
    except:
        true = results_test['TRUE']
        
    plot_confusion_matrix(true, results_test['pred_rules'], ["Low", "Moderate", "High"],
                          "Rule-Based Confusion Matrix", os.path.join(test_rep_dir, "Rule-Based Confusion Matrix Full Results.png"))
    plot_confusion_matrix(true, results_test['pred_few'], ["Low", "Moderate", "High"],
                          "Few Shot Confusion Matrix", os.path.join(test_rep_dir, "Few Shot Confusion Matrix Full Results.png"))
    plot_confusion_matrix(true, results_test['pred_zero'], ["Low", "Moderate", "High"],
                          "Zero Shot Confusion Matrix", os.path.join(test_rep_dir, "Zero Shot Confusion Matrix Full Results.png"))
    
   


if __name__ == "__main__":
    try:
        start = datetime.now()
        time_stamp('start')
        # run_pipeline('o', 10)
        # run_pipeline('c', 10)
        run_pipeline('a', 10)
        run_pipeline('n', 10)
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
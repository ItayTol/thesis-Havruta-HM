# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:00:10 2025

@author: User
"""


import openai
import numpy as np
# import re
from plyer import notification
import os
import pandas as pd
import json
from datetime import datetime
from havruta_HM_thesis_prompts_3_classes import load_trait
from havruta_HM_thesis_classification import rank_rules_and_classify
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from havruta_HM_thesis_classification_prep import evaluate_zero_shot, evaluate_few_shot
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss, matthews_corrcoef, cohen_kappa_score,
    top_k_accuracy_score, roc_auc_score)
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

openai.api_key = os.getenv('OPENAI_API_KEY')
pd.set_option('display.max_colwidth', None)  # Show full text in cells

    # ---- small helpers ---------------------------------------------------------

# Load dataset (social media posts + true labels)
def load_posts_and_trait_true_label(personality_trait: str, num_classes):
    df = pd.read_excel("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/133_participants_data.xlsx",
                       sheet_name=f"summary_true_labels_{num_classes}_classes")
  
    filter_condition = df['USE FOR'] == 'final test'
    return df.loc[filter_condition, :]

# Load dataset (social media posts + true labels)
def load_rule_set(personality_trait: str, version, psych: str = None):
    if psych:
        df = pd.read_excel(fr"C:\Users\User\OneDrive\Desktop\Masters\thesis\DATA\rules_v{version}_{psych}.xlsx",
                           sheet_name=load_trait(personality_trait))
    else:
        df = pd.read_excel(fr"C:\Users\User\OneDrive\Desktop\Masters\thesis\DATA\rules_v{version}.xlsx",
                           sheet_name=load_trait(personality_trait))
        
    # 'records' orientation (list of dictionaries, each representing a row)
    json_records = df.to_json(orient='records', indent = 4)
    list_of_rules = json.loads(json_records)
    return list_of_rules

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

def build_few_shot_prompt(personality_trait: str, true_label_type: str, k: int = 5) -> str:
    """Builds a short few-shot block from the training slice (UTF-8 safe)."""
    df = pd.read_excel("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/133_participants_data.xlsx",
                       sheet_name="summary_true_labels_3_classes")
    filter_condition = df['USE FOR'] == 'rule generation'
    filtered_df = df.loc[filter_condition, :]
    records_for_few_shot = filtered_df.sample(k)
    lines = []

    if true_label_type == 'percentiles':
        personality_trait_true_label = f'{personality_trait} by percentiles'
      
    elif true_label_type == 'fixed ranges' :
        personality_trait_true_label = f'{personality_trait} by fixed ranges'

    for i, row in records_for_few_shot.iterrows():
        lines.append(
            f"Example {i+1}\n"
            f"Texts:\n1. {row['post1']}\n2. {row['post2']}\n"
            f"{load_trait(personality_trait)} level: {row[personality_trait_true_label]}\n"
        )
    return "\n".join(lines)

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def _json_dump_safely(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def run_pipeline(
    version: str,
    true_label_type: str,
    outer_reps: int = 3,
    num_classes: int = 3,
    psych: str = None
):
    
    print('version -', version, '; psych -', psych, '; num_classes -', num_classes, '; true_label_type -', true_label_type)
    summary_results_test = []
    results_test_list = []
    for personality_trait in ['e']:
    # ['a', 'n', 'c', 'o', 'e']:
        final_test_dir_root: str = r"C:/Users/User/OneDrive/Desktop/Masters/thesis/final test"
        # --- results folders ---
        trait_folder = f"{load_trait(personality_trait)}"
        base_path = os.path.join(
            final_test_dir_root,
            f"run_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')} - Version{version}_{psych}_{num_classes}_{true_label_type}",
            trait_folder
        )
        os.makedirs(base_path, exist_ok=True)

        subdirs = {
            "few_shot_examples": os.path.join(base_path, "_meta"),
        }
        for p in subdirs.values():
            os.makedirs(p, exist_ok=True)

      
        final_test = load_posts_and_trait_true_label(personality_trait, num_classes)  
        rules = load_rule_set(personality_trait, version, psych)

        for outer_rep in range(1, outer_reps + 1):
            few_shot_examples = build_few_shot_prompt(personality_trait, true_label_type)
            
            # # Store few-shot block (helps tracing)
            with open(os.path.join(subdirs["few_shot_examples"], f"FEW_SHOT_EXAMPLES_OUTER{outer_rep}.txt"), "w", encoding="utf-8") as f:
                if isinstance(few_shot_examples, (dict, list)):
                    f.write(json.dumps(few_shot_examples, ensure_ascii=False, indent=2))
                else:
                    f.write(str(few_shot_examples))    

            # (a) Rule-based
            preds_rule = []
            reses_rule = []
            expla_rule = []
            ok_rule = []
            for _, row in final_test.iterrows():
                res = rank_rules_and_classify(personality_trait, row["post1"], row["post2"], rules)
                indented_res = json.dumps(res, ensure_ascii=False, indent=2)
                reses_rule.append(indented_res)
                try:
                    preds_rule.append(res["trait_classification"]["classified_level"])
                except:
                    print(res)

                expla_rule.append(res["trait_classification"]["justification"])
            if true_label_type == 'percentiles':
                personality_trait_true_label = f'{personality_trait} by percentiles'

            elif true_label_type == 'fixed ranges' :
                personality_trait_true_label = f'{personality_trait} by fixed ranges'
                
            else:
                print('Choose true labels by percentiles or fixed ranges')
                break
            
            ok_rule = [int(pred == true) for pred, true in zip(preds_rule, final_test[personality_trait_true_label])]
            acc_rule = np.mean([pred == true for pred, true in zip(preds_rule, final_test[personality_trait_true_label])])
            macro_f1_rule = f1_score(final_test[personality_trait_true_label], preds_rule, average="macro")
            df_res_rule = pd.DataFrame({'rep': outer_rep, "p": final_test.get("p"), 
                                        "post1": final_test["post1"], "post2": final_test["post2"],
                                        'true': final_test[personality_trait_true_label], 
                                        "pred_rules": preds_rule, 'explanation_rules':expla_rule,
                                        'res_rules': reses_rule, 'accuracy_rules': ok_rule})
           
            # df_res_rule.to_csv(os.path.join(test_rep_dir, f"TEST-OUTER{outer_rep}.csv"), index=False, encoding="utf-8")
            summary_results_test.append({"outer_rep": outer_rep, 
                                        "method": "rules_based", 
                                        "acc": acc_rule,
                                        "macro f1": macro_f1_rule})
            # (b) Zero-shot
            preds_zero = []
            reses_zero = []
            expla_zero = []
            ok_zero = []
            for i, row in final_test.iterrows():
                res = evaluate_zero_shot(personality_trait, row["post1"], row["post2"])
                indented_res = json.dumps(res, ensure_ascii=False, indent=2)
                reses_zero.append(indented_res)
                preds_zero.append(res["level"])
                expla_zero.append(res['explanation'])
            acc_zero = np.mean([pred == true for pred, true in zip(preds_zero, final_test[personality_trait_true_label])])
            macro_f1_zero = f1_score(final_test[personality_trait_true_label], preds_zero, average="macro")

            ok_zero = [int(pred == true) for pred, true in zip(preds_zero, final_test[personality_trait_true_label])]
            df_res_zero = pd.DataFrame({"p": final_test.get("p"), "pred_zero": preds_zero, 
                                        'explanation_zero': expla_zero, "accuracy_zero": ok_zero})
            # df_res_zero.to_csv(os.path.join(test_rep_dir, "TEST-OUTER{0}-ZERO.csv".format(outer_rep)), index=False, encoding="utf-8")
            summary_results_test.append({"outer_rep": outer_rep, "method": "zero_shot", 
                                         "acc": float(acc_zero), "macro f1": macro_f1_zero})
    
            # (c) Few-shot (using DEV-derived examples)
            preds_few = []
            reses_few = []
            expla_few = []
            ok_few = []
           
            for _, row in final_test.iterrows():
                res = evaluate_few_shot(personality_trait, row["post1"], row["post2"], few_shot=few_shot_examples)
                indented_res = json.dumps(res, ensure_ascii=False, indent=2)
                reses_few.append(indented_res)
                preds_few.append(res['level'])
                expla_few.append(res['explanation'])
            acc_few = np.mean([pred == true for pred, true in zip(preds_few, final_test[personality_trait_true_label])])
            macro_f1_few = f1_score(final_test[personality_trait_true_label], preds_few, average="macro")

            ok_few = [int(pred == true) for pred, true in zip(preds_few, final_test[personality_trait_true_label])]
            df_res_few = pd.DataFrame({"p": final_test.get("p"), "pred_few": preds_few, 
                                       'explanation_few':expla_few, "accuracy_few": ok_few})
            # df_res_few.to_csv(os.path.join(test_rep_dir, "TEST-OUTER{0}-FEW.csv".format(outer_rep)), index=False, encoding="utf-8")
            summary_results_test.append({"outer_rep": outer_rep, "method": "five_shot", 
                                          "acc": float(acc_few), "macro f1": macro_f1_few})
            
            rule_zero_merged = pd.merge(df_res_rule, df_res_zero, on = "p", how = "inner")
            results_rep_test = pd.merge(rule_zero_merged, df_res_few, on = "p", how = "inner")
            results_test_list.append(results_rep_test)
            # Save outer rep summary
            results_rep_test.to_csv(os.path.join(base_path, f"TEST-FULL-Rep{outer_rep}-RESULTS.csv"), index=False, encoding="utf-8-sig")

            df_rep_summary = pd.DataFrame(summary_results_test)
            df_rep_summary.to_csv(os.path.join(base_path, f"TEST-Rep{outer_rep}-SUMMARY.csv"),  index=False, encoding="utf-8-sig")
            
            try:
                true = results_rep_test['true']
            except:
                true = results_rep_test['TRUE']

            plot_confusion_matrix(true, results_rep_test['pred_rules'], ["Low", "Moderate", "High"],
                                  f"Rule-Based Confusion Matrix rep {outer_rep}/{outer_reps}", os.path.join(base_path, f"Rule-Based Confusion Matrix Full Results rep ({outer_rep} out of {outer_reps}).png"))
            plot_confusion_matrix(true, results_rep_test['pred_few'], ["Low", "Moderate", "High"],
                                  f"Few Shot Confusion Matrix rep {outer_rep}/{outer_reps}", os.path.join(base_path, f"Few Shot Confusion Matrix Full Results rep ({outer_rep} out of {outer_reps}).png"))
            plot_confusion_matrix(true, results_rep_test['pred_zero'], ["Low", "Moderate", "High"],
                                  f"Zero Shot Confusion Matrix rep {outer_rep}/{outer_reps}", os.path.join(base_path, f"Zero Shot Confusion Matrix Full Results rep ({outer_rep} out of {outer_reps}).png"))


    # --- Final summary ---
    try:
        results_test = pd.concat(results_test_list) 
        print('pd.concat(results_test_list) worked')
    except:
        results_test = results_rep_test

    results_test.to_csv(os.path.join(base_path, "TEST-FULL-RESULTS.csv"), index=False, encoding="utf-8-sig")
    
    try:
        true = results_test['true']
    except:
        true = results_test['TRUE']

    plot_confusion_matrix(true, results_test['pred_rules'], ["Low", "Moderate", "High"],
                          "Rule-Based Confusion Matrix", os.path.join(base_path, "Rule-Based Confusion Matrix Full Results.png"))
    plot_confusion_matrix(true, results_test['pred_few'], ["Low", "Moderate", "High"],
                          "Few Shot Confusion Matrix", os.path.join(base_path, "Few Shot Confusion Matrix Full Results.png"))
    plot_confusion_matrix(true, results_test['pred_zero'], ["Low", "Moderate", "High"],
                          "Zero Shot Confusion Matrix", os.path.join(base_path, "Zero Shot Confusion Matrix Full Results.png"))

if __name__ == "__main__":
    try:
        start = datetime.now()
        time_stamp('start')

        run_pipeline(version = 1, true_label_type = 'fixed ranges', psych='zohar')
        run_pipeline(version = 2, true_label_type = 'fixed ranges', psych='zohar')

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
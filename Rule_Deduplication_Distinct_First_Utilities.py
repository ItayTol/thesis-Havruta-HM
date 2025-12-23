# import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import pandas as pd

def load_all_rules_from_rules_dir(base_dir):
    """
    Traverse subfolders inside base_dir, load all JSON files,
    and collect all rule dictionaries into a single list.
    """
    all_rules = []

    for subdir in os.listdir(base_dir):
        full_subdir = os.path.join(base_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue  # skip non-folder files

        for filename in os.listdir(full_subdir):
            if filename.endswith(".json"):
                filepath = os.path.join(full_subdir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        rules = json.load(f)
                        if isinstance(rules, list):
                            all_rules.extend(rules)
                    except json.JSONDecodeError as e:
                        print(f"❌ Error reading {filepath}: {e}")
    
    all_rules_df = pd.DataFrame(all_rules)
    # all_rules_df.to_csv(r"C:\Users\User\OneDrive\Desktop\Masters\thesis\Results\all_rules_df_n.csv")
    return all_rules_df


def summarize_and_merge_rules(results_path_csv):
    """
    Reads:
      - results_csv: file containing rule match results and predictions

    Returns:
      A merged DataFrame of rules with stats and text fields.
    """

    # --- 1. Load model results ---
    df = pd.read_csv(results_path_csv)
    rule_stats = {}

    for _, row in df.iterrows():
        try:
            rule_results = json.loads(row["res_rules"])
            true_label = str(row["true"]).strip().lower()
            pred_label = str(row["pred_rules"]).strip().lower()
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue

        for rm in rule_results.get("rule_matches", []):
            if rm.get("rule_applies", False):
                name = rm.get("rule_name")
                if name not in rule_stats:
                    rule_stats[name] = {"applied": 0, "correct": 0}
                rule_stats[name]["applied"] += 1
                if pred_label == true_label:
                    rule_stats[name]["correct"] += 1
                    
    # --- 2. Convert stats to DataFrame ---
    stats_df = pd.DataFrame([
        {
            "rule_name": name,
            "applied_count": v["applied"],
            "correct_matches": v["correct"],
            "alignment_rate": round(v["correct"] / v["applied"], 3)
            if v["applied"] > 0 else 0
        }
        for name, v in rule_stats.items()
    ])        
    filtered_stats_df = stats_df#.query('applied_count>=5 and alignment_rate>=0.3').reset_index(drop=True)
    return filtered_stats_df

import os

if __name__ == "__main__":
    
    # ----------------
    results_path_v0 = r"C:\Users\User\OneDrive\Desktop\Masters\thesis\final test\run_2025-11-10_21-36-17 - Version0_None\extraversion\TEST-FULL-RESULTS.csv"
    df_results_v0 = pd.read_csv(results_path_v0)
    macro_f1_rules_v0 = f1_score(df_results_v0['true'], df_results_v0['pred_rules'], average = 'macro')
    macro_f1_zero_v0 = f1_score(df_results_v0['true'], df_results_v0['pred_zero'], average = 'macro')
    macro_f1_few_v0 = f1_score(df_results_v0['true'], df_results_v0['pred_few'], average = 'macro')
    
    print('macro f1-score rules method (v0):', macro_f1_rules_v0)
    print('macro f1-score zero shot method (v0):', macro_f1_zero_v0)
    print('macro f1-score few shot (v0):', macro_f1_few_v0)
    print('')
    
    # -----------------
    results_path_v1_zohar = r"C:\Users\User\OneDrive\Desktop\Masters\thesis\final test\run_2025-12-01_20-08-03 - Version1_zohar\extraversion\TEST-FULL-RESULTS.csv"
    df_results_v1_zohar = pd.read_csv(results_path_v1_zohar)
    macro_f1_rules_v1_zohar = f1_score(df_results_v1_zohar['true'], df_results_v1_zohar['pred_rules'], average = 'macro')
    macro_f1_zero_v1_zohar = f1_score(df_results_v1_zohar['true'], df_results_v1_zohar['pred_zero'], average = 'macro')
    macro_f1_few_v1_zohar = f1_score(df_results_v1_zohar['true'], df_results_v1_zohar['pred_few'], average = 'macro')
    
    print('macro f1-score rules method (v1 zohar):', macro_f1_rules_v1_zohar)
    print('macro f1-score zero shot method (v1 zohar):', macro_f1_zero_v1_zohar)
    print('macro f1-score few shot (v1 zohar):', macro_f1_few_v1_zohar)
    print('')
   # ------------------- 
    results_path_v2_zohar = r"C:\Users\User\OneDrive\Desktop\Masters\thesis\final test\run_2025-12-01_18-56-04 - Version2_zohar\extraversion\TEST-FULL-RESULTS.csv"
    df_results_v2_zohar = pd.read_csv(results_path_v2_zohar)
    macro_f1_rules_v2_zohar = f1_score(df_results_v2_zohar['true'], df_results_v2_zohar['pred_rules'], average = 'macro')
    macro_f1_zero_v2_zohar = f1_score(df_results_v2_zohar['true'], df_results_v2_zohar['pred_zero'], average = 'macro')
    macro_f1_few_v2_zohar = f1_score(df_results_v2_zohar['true'], df_results_v2_zohar['pred_few'], average = 'macro')
    
    print('macro f1-score rules method (v2 zohar):', macro_f1_rules_v2_zohar)
    print('macro f1-score zero shot method (v2 zohar):', macro_f1_zero_v2_zohar)
    print('macro f1-score few shot (v2 zohar):', macro_f1_few_v2_zohar)
    
   
    rules_v1_zohar_path = r"C:\Users\User\OneDrive\Desktop\Masters\thesis\DATA\rules_v1_zohar.xlsx"
    rules_v1_zohar_df = pd.read_excel(rules_v1_zohar_path)
    filtered_stats_df = summarize_and_merge_rules(results_path_v1_zohar)
    x = pd.merge(rules_v1_zohar_df, filtered_stats_df, left_on='rule_name', right_on='rule_name', how = 'inner').reset_index(drop=True)
    x.to_csv(r'C:\Users\User\OneDrive\Desktop\Masters\thesis\final test\rules_v1_extraversion_zohar_with_coverage_scores.csv', encoding="utf-8-sig")
    print("✅ Fin version 1")
        
    rules_v2_zohar_path = r"C:\Users\User\OneDrive\Desktop\Masters\thesis\DATA\rules_v2_zohar.xlsx"
    rules_v2_zohar_df = pd.read_excel(rules_v2_zohar_path)
    filtered_stats_df = summarize_and_merge_rules(results_path_v2_zohar)
    x = pd.merge(rules_v2_zohar_df, filtered_stats_df, left_on='rule_name', right_on='rule_name', how = 'inner').reset_index(drop=True)
    x.to_csv(r'C:\Users\User\OneDrive\Desktop\Masters\thesis\final test\rules_v2_extraversion_zohar_with_coverage_scores.csv', encoding="utf-8-sig")
    print("✅ Fin version 2")
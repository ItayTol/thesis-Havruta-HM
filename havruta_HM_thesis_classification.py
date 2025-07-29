# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:57:25 2025

@author: User
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from havruta_HM_thesis_classification_prep  import evaluate_few_shot, evaluate_zero_shot, rank_rules_and_classify
from collections import Counter

def classifier_zero_and_few_shot(personality_trait, posts_df, extra_prompt_few_shot, rep, fold):    
    results = []
    true_labels = []
    for _, row in posts_df.iterrows():
        p = row['p']
        post1, post2 = row['post1'], row['post2']
        true_label = row[personality_trait]
        true_labels.append(true_label)
        result_pred_few = evaluate_few_shot(personality_trait, post1, post2, extra_prompt_few_shot)
        result_pred_zero = evaluate_zero_shot(personality_trait, post1, post2)
        
        result_score_few = int(true_label == result_pred_few)
        result_score_zero = int(true_label == result_pred_zero)
        results.append({
            "p": p,
            "true_label": true_label,
            "few_shot_pred": result_pred_few,
            "few_shot_score": result_score_few,
            "zero_shot_pred": result_pred_zero,
            "zero_shot_score": result_score_zero,            
        })
        
    return pd.DataFrame(results)

import json
# In each iteration, go over and classify  test set all
def classifier_using_rules(personality_trait, posts_df, rules, iteration=None):    
    results_for_rules = []
    list_rules_pred, list_zero_pred = [], []
    true_labels = []
    # dict_class_to_code = {'Unknown':0, 'Low': 1, 'Moderate': 2, 'High': 3}

    for _, row in posts_df.iterrows():
        p = row['p']
        post1, post2 = row['post1'], row['post2']
        true_label = row[personality_trait]    
        true_labels.append(true_label)
        
        result_pred_rules = rank_rules_and_classify(personality_trait, post1, post2, rules)
        trait_classification = result_pred_rules['trait_classification']
        classified_level = trait_classification['classified_level']
        list_rules_pred.append(classified_level)
       
        sample_zero_pred = evaluate_zero_shot(personality_trait, post1, post2)
        list_zero_pred.append(sample_zero_pred)
        # Parse GPT JSON output (already formatted to spec)
        
        matched_rules_names = [entry['rule_name'] for entry in result_pred_rules['rule_matches'] if entry['relevance_rating'] > 0]
        matched_rules = [entry['behavior_rule'] for entry in rules if entry['rule_name'] in matched_rules_names]
        supporting_examples = [entry['supporting_examples'] for entry in rules if entry['rule_name'] in matched_rules_names]
        psychological_justification = [entry['psychological_justification'] for entry in rules if entry['rule_name'] in matched_rules_names]
        edge_cases = [entry['edge_cases'] for entry in rules if entry['rule_name'] in matched_rules_names]
        linguistic_indicators = [entry['linguistic_indicators'] for entry in rules if entry['rule_name'] in matched_rules_names]

        results_for_rules.append({
            "p": p,
            "post1":post1,
            "post2":post2,
            "true_label": true_label,
            "matched_rules": matched_rules,
            "supporting_examples": supporting_examples,
            "psychological_justification": psychological_justification,
            "linguistic_indicators": linguistic_indicators,
            "edge_cases": edge_cases,
            "pred_zero_shot": sample_zero_pred,
            "pred_rule_based": classified_level,
        })    
    results_df = pd.DataFrame(results_for_rules)
    score_rules = [1 if pred == true else 0 for pred, true in zip(list_rules_pred, true_labels)]
    score_zero = [1 if pred == true else 0 for pred, true in zip(list_zero_pred, true_labels)]
    results_df['score_rules'] = score_rules 
    results_df['score_zero'] = score_zero 
    
    
    # pred_label_code = list(map(lambda x: dict_class_to_code[x], list_rules_pred))
    # true_label_code = list(map(lambda x: dict_class_to_code[x], true_labels))
    
    # print(f'Rules accuracy is {np.round(np.mean(score_rules), 3)}\nRules Mean Absolute Error is {np.round(np.mean(mae_rules), 3)}')
    return results_df

# Recursive filtering function
def recursive_rule_filtering(personality_trait, posts_df, rules, rep, fold,
                              min_matches, correct_ratio_threshold, max_iterations, path, directory_names):

    iteration_logs = []
    current_rules = rules.copy()
    removed_rules = []
    dict_iterations_result_df = {}
    
    dict_class_to_code = {'Unknown': -10, 'Low': 1, 'Moderate': 2, 'High': 3}

    for iteration in range(1, max_iterations + 1):
        # Remove previously flagged rules
        current_rules = [r for r in current_rules if r not in removed_rules]
        print(f"\nIteration {iteration}\nNumber of rules being used = {len(current_rules)}")
        for r in current_rules:
            print(r, '\n')
        # Apply classifier with the current set of rules
        # Returns DataFrame with columns: p, post1, post2, true_label, 
        # matched_rules, matched_behaviours, sample_rules_pred
        results_df = classifier_using_rules(personality_trait, posts_df, current_rules, iteration)
        iteration_accuracy = np.round(np.mean(results_df['score_rules']), 3)
        true_label_code = results_df["true_label"].map(dict_class_to_code)
        pred_label_code = results_df["pred_rule_based"].map(dict_class_to_code)
        iteration_mae = np.round(mean_absolute_error(true_label_code, pred_label_code), 3)
        dict_iterations_result_df.update({iteration: (iteration_accuracy, iteration_mae, results_df)})
        # results_df.to_csv(f'{path}{directory_names[2]}/results rep {rep} fold {fold} iteration {iteration}.csv', index=False)
        # Accumulate stats per rule
        rule_stats = defaultdict(lambda: {"matched_p": [], "true_labels": [], "pred_rule_based": []})
        for _, row in results_df.iterrows():
            for rule in row["matched_rules"]:
                if rule in current_rules:  # Avoid counting already removed ones
                    rule_stats[rule]["matched_p"].append(row["p"])
                    rule_stats[rule]["true_labels"].append(row["true_label"])
                    rule_stats[rule]["pred_rule_based"].append(row["pred_rule_based"])
                    
        # with open(f'{path}{directory_names[3]}/Rule Stats rep {rep} fold {fold} iteration {iteration}.txt', 'w') as f:
        #     for item in rule_stats.items():
        #         f.write(str(item) + '\n')
            
        # Analyze each rule's accuracy
        to_remove = []
        for rule, info in rule_stats.items():
            print('\n', rule)
            # match_count => The number of participants matching with the rule
            match_count = len(set(info["matched_p"]))
            
            correct_matches = sum(1 for true_label, pred in zip(info["true_labels"], info['pred_rule_based']) if true_label == pred)
            # correct_ratio = Out of the participants who matched, the ratio between correct predications to overall predications 
            correct_ratio = correct_matches / match_count if match_count > 0 else 0
    
            if match_count >= min_matches and correct_ratio < correct_ratio_threshold :                
                print('\n     - Rule -', rule, ', Appends to removals')
                print(f'     - match_count ({match_count}) ">=" min_matches ({min_matches}) AND correct_ratio ({correct_ratio}) < correct_ratio_threshold ({correct_ratio_threshold})')
                to_remove.append(rule)
            print('\n\n')
        if not to_remove:
            iteration_logs.append({
                "iteration": iteration,
                "removed_rules": [],
                "rules_remaining": len(current_rules),
                "iteration_accuracy": iteration_accuracy,
                "iteration_mae": iteration_mae
            })
            break  # Nothing left to prune

        removed_rules.extend(to_remove)
        iteration_logs.append({
            "iteration": iteration,
            "removed_rules": to_remove,
            "rules_remaining": len(current_rules) - len(to_remove),
            "iteration_accuracy": iteration_accuracy,
            "iteration_mae": iteration_mae
        })
        # if not iteration_logs:
            
    # Find the iteration with the highest accuracy
    best_iter, (best_acc, best_mae, best_df) = max(dict_iterations_result_df.items(), key=lambda x: x[1][0])
    pd.DataFrame(iteration_logs).to_csv(f'{path}{directory_names[0]}/log rep {rep} fold {fold}.csv', index=False)
    return best_df
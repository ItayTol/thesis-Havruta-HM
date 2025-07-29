import openai
# import re
from plyer import notification
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from datetime import datetime
from sklearn.model_selection import KFold
from havruta_HM_thesis_prompts_3_classes import load_trait
from havruta_HM_thesis_load_rules_3_classes import load_rules_with_gpt
from havruta_HM_thesis_gpt_response_3_classes import categorize_rules_with_gpt, get_explanation_zero_shot
from havruta_HM_thesis_recursion_3_classes import  classify_zero_and_few_shot, recursive_rule_filtering, classifier_using_rules

openai.api_key = os.getenv('OPENAI_API_KEY')
pd.set_option('display.max_colwidth', None)  # Show full text in cells

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------

# Load dataset (social media posts + true labels)
def load_posts_and_trait_true_label(personality_trait: str):
    df = pd.read_csv("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/true_labels_3_classes.csv", encoding='cp1252')
    return df[['p', 'post1', 'post2', f'{personality_trait}']]

        # ------------ AND CLAUSE Changed to OR CLAUSE--------------
def check_partial_agreement_and_match(psych_labels, true_label, agreement_threshold=0):
    """
    Checks if there is partial agreement among psychologist labels, 
    and whether it matches the true trait label.
    
    Parameters:
    - psych_labels: list of 2 labels from psychologists (e.g., ["High", "High", "Moderate"])
    - true_label: the true trait label (e.g., "High")
    - agreement_threshold: minimum number of psychologists who must agree on a label
    
    Returns:
    - has_partial_agreement: True/False
    - agreed_label: the label agreed upon (if any)
    - matches_true_label: True/False (if agreement exists)
    """
    label_counts = Counter(psych_labels)
    most_common_label, count = label_counts.most_common(1)[0]

    if count >= agreement_threshold or most_common_label == true_label:
        return True
    return False

def get_agrees_partialy_or_full_with_aligning(personality_trait):  
    agrees_partialy_or_full_with_aligning = []
    df = pd.read_excel('C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/rules_made_by_gpt.xlsx', personality_trait)  
    for i in range(0, len(df), 3):
        psych_labels = list(df.iloc[i:i+3]['psych rating'])
        true_label = df.iloc[i]['true rating']
        if check_partial_agreement_and_match(psych_labels, true_label):
            s = df.at[i, 'p']
            agrees_partialy_or_full_with_aligning.append(s)
    return agrees_partialy_or_full_with_aligning

# Define and save the identify_problematic_rules function again
#In use by 'main_seperating_matched_rules_and_classification' and 'main_matched_rules'

def identify_problematic_rules(df, match_threshold=8, dominance_threshold=0.9):        
    """
    Identifies problematic rules based on:
    - How often they were matched (match_threshold)
    - Whether they overwhelmingly push to one label (dominance_threshold)

    Returns a DataFrame of problematic rules.
    """
    rule_stats = {}

    for _, row in df.iterrows():
        for rule in row["matched_rules"]:
            if rule not in rule_stats:
                rule_stats[rule] = {"match_count": 0, "high": 0, "moderate": 0, "low": 0}
            rule_stats[rule]["match_count"] += 1
            if row["predicted_label"] == "High":
                rule_stats[rule]["high"] += 1
            elif row["predicted_label"] == "Moderate":
                rule_stats[rule]["moderate"] += 1
            elif row["predicted_label"] == "Low":
                rule_stats[rule]["low"] += 1

    # Convert to DataFrame and calculate dominance ratio
    problematic_rules = []
    for rule, stats in rule_stats.items():
        total = stats["match_count"]
        for label in ["high", "moderate", "low"]:
            ratio = stats[label] / total if total else 0
            if total >= match_threshold and ratio >= dominance_threshold:
                problematic_rules.append({
                    "Rule": rule,
                    "Total_Matches": total,
                    "Dominant_Label": label.capitalize(),
                    "Dominance_Ratio": round(ratio, 2)
                })
            break  # No need to check other labels
    return problematic_rules

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


#######################################

from sklearn.utils import shuffle

def main_recursive(personality_trait: str, num_reps: int, start_txt):
    participants = load_posts_and_trait_true_label(personality_trait)

    dict_class_to_code = {'Unknown':0, 'Low': 1, 'Moderate': 2, 'High': 3}
    
    # Specify the directory name
    path_trait = f'C:/Users/User/OneDrive/Desktop/Masters/thesis/code 3 classes/{load_trait(personality_trait)}/'
    run_folder = f'run_{start_txt}/'
    os.mkdir(''.join(path_trait + run_folder))
    path = path_trait + run_folder
    directory_names = ["logs", "classification_results", "validation set", "Top Rules", "rule_to_category"] 
    for directory_name in directory_names:
        # Create the directory
        try:
            os.mkdir(''.join(path + directory_name))
            # print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError: 
            pass
            # print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    all_reps_accuracies_rules, all_reps_maes_rules = [], []
    all_reps_accuracies_few, all_reps_maes_few = [], []
    all_reps_accuracies_zero, all_reps_maes_zero = [], []
    final_results = []
    for rep in range(1, num_reps + 1):
        print('----------Replication', rep, 'Beginess----------')

        rules = []
        start_rep = datetime.now()
        rep_results_test_rules, rep_results_test_zero_few_shot = [], []
        rep_top_rules = []
        
        
        # Shuffle and reset index
        participants = shuffle(participants, random_state=42).reset_index(drop=True)
        
        # Split into 3 equal parts of 20 samples each
        num_folds = 3
        folds = [participants[i*20:(i+1)*20] for i in range(num_folds)]
        
        splits = []
        for fold in range(num_folds):
            test = folds[fold].iloc[:10]
            train = folds[fold].iloc[10:]
            val = pd.concat([folds[j] for j in range(num_folds) if j != fold], ignore_index=True)
            splits.append({'fold': fold+1, 'train': train, 'val': val, 'test': test})

            print('\n     ----Fold', fold+1, f'Beggines (rep {rep})----')
            # Get effective training data (rules and few shot from same train dataset)
            train_participants = train['p'].tolist()
            # agrees_partialy_or_full_with_aligning = get_agrees_partialy_or_full_with_aligning(personality_trait)
            participants_for_rules = list(set(train_participants))# & set(agrees_partialy_or_full_with_aligning))  
            # Prepare rules using training data  
            rules = load_rules_with_gpt(personality_trait, participants_for_rules)
    
            for r in rules:
                print(r, '\n')
                
            return None
            # Prepare dict containing key = rule, item = category
            rule_to_category = categorize_rules_with_gpt(personality_trait, rules)
            
            # Build few-shot prompt using training data
            extra_prompt_few_shot = ""
            for i, row in train.sample(5).iterrows():
                extra_prompt_few_shot += (
                    f"Example {i+1}\nInput:\n1. {row['post1']}\n2. {row['post2']}\n"
                    f"Output: {row[personality_trait]}\n\n")
# -------------------------------------------------------------------------------------------------
            # Get most helpful rules using validation data
            # print('min_matches:', len(val['p'])/2)
            result_val_rules = recursive_rule_filtering(
                personality_trait, val, rules, rule_to_category, rep, fold, 
                min_matches = len(val['p'])/2, correct_ratio_threshold = 0.67, max_iterations = 4, 
                path = path, directory_names = directory_names)     
            result_val_rules.to_csv(f"{path}{directory_names[2]}/classification validation set results rep {rep} fold {fold}.csv", index=False)
            
            matched_rules_in_val_data = result_val_rules.loc[(result_val_rules['score_zero'].astype(str) == '0') & (result_val_rules['score_rules'].astype(str) == '1'), 'matched_rules'].to_list()
            rules_flattened = []
            for rules_matched_to_sample in matched_rules_in_val_data:
                for rule in rules_matched_to_sample:
                    rules_flattened.append(rule)
            counts = Counter(rules_flattened)
            top_rules = [key for key, _ in counts.most_common()]
            for rule in top_rules:
                rep_top_rules.append(rule)
            print('\nTop Rules:')
            for r in top_rules:
                print(r, '\n')
            
# -------------------------------------------------------------------------------------------------
            # Evaluate top rules on test data             
            top_rule_to_category = categorize_rules_with_gpt(personality_trait, top_rules)     
            result_test_rules = classifier_using_rules(personality_trait, test, top_rules, top_rule_to_category)
            result_test_rules['fold'] = fold
            rep_results_test_rules.append(result_test_rules) 

            
            # Evaluate few and zero shot on test data             
            result_test_zero_and_few_shot = classify_zero_and_few_shot(personality_trait, 
                                                                         test, extra_prompt_few_shot, rep, fold) 
            rep_results_test_zero_few_shot.append(result_test_zero_and_few_shot)
# -------------------------------------------------------------------------------------------------
        # Concatenate folds and save per rep        
        with open(f'{path}{directory_names[3]}/Top Rules rep {rep}.txt', 'w') as f:
            for item in rep_top_rules:
                f.write(str(item) + '\n')
                
        rep_test_rules = pd.concat(rep_results_test_rules, ignore_index=True)
        rep_test_zero_few_shot = pd.concat(rep_results_test_zero_few_shot, ignore_index=True)
        
        rep_test_rules["matched_rules"] = rep_test_rules["matched_rules"].apply(lambda r: "; ".join(r) if isinstance(r, list) else str(r))
        rep_test_rules["matched_behaviours"] = rep_test_rules["matched_behaviours"].apply(lambda r: "; ".join(r) if isinstance(r, list) else str(r))
        rep_test_rules["rep"] = rep
        rep_test_rules["score_rules"] = [int(pred == true) for (true, pred) in zip(rep_test_rules['true_label'], rep_test_rules['pred_rule_based'])]
        rep_accuracy_rules = np.round(accuracy_score(rep_test_rules['true_label'], rep_test_rules['pred_rule_based']), 3)
        rep_mae_rules = np.round(mean_absolute_error(rep_test_rules['true_label'].map(dict_class_to_code), 
                                      rep_test_rules['pred_rule_based'].map(dict_class_to_code)), 3)
        
        rep_accuracy_few = np.round(accuracy_score(rep_test_zero_few_shot['true_label'], rep_test_zero_few_shot['few_shot_pred']), 3)
        rep_mae_few = np.round(mean_absolute_error(rep_test_zero_few_shot['true_label'].map(dict_class_to_code), 
                                      rep_test_zero_few_shot['few_shot_pred'].map(dict_class_to_code)), 3)
        rep_accuracy_zero = np.round(accuracy_score(rep_test_zero_few_shot['true_label'], rep_test_zero_few_shot['zero_shot_pred']), 3)
        rep_mae_zero = np.round(mean_absolute_error(rep_test_zero_few_shot['true_label'].map(dict_class_to_code), 
                                      rep_test_zero_few_shot['zero_shot_pred'].map(dict_class_to_code)), 3)
        
        try:
            merged_rep = rep_test_rules.merge(rep_test_zero_few_shot, on=['p', 'true_label'], how='inner')
            merged_rep.to_csv(f"{path}{directory_names[1]}/classification test results rep {rep} combined.csv", index=False)
        except:
            print('Rep - The join between rules and zero few shot has failed!')

        all_reps_accuracies_rules.append(rep_accuracy_rules)
        all_reps_maes_rules.append(rep_mae_rules)
        
        end_rep = datetime.now() - start_rep
        total_minutes = int(end_rep.total_seconds() / 60)
        print(f'#### Rules Rep {rep}\nDuration: {total_minutes} minutes\nAccuracy: {rep_accuracy_rules}, \nMean Absolute Error: {rep_mae_rules}\n')
  
        # save per rep        
        print(f'#### Few Rep {rep}\nAccuracy: {rep_accuracy_few}, \nMean Absolute Error: {rep_mae_few}\n')
        print(f'#### Zero Rep {rep}\nAccuracy: {rep_accuracy_zero}, \nMean Absolute Error: {rep_mae_zero}')

        final_results.append(merged_rep)
        all_reps_accuracies_few.append(rep_accuracy_few)
        all_reps_maes_few.append(rep_mae_few)

        all_reps_accuracies_zero.append(rep_accuracy_zero)
        all_reps_maes_zero.append(rep_mae_zero)


        rep_test_rules.to_csv(f"{path}{directory_names[1]}/classification results rules rep {rep}.csv", index=False)
        rep_test_zero_few_shot.to_csv(f"{path}{directory_names[1]}/classification results zero few shot rep {rep}.csv", index=False)
        print('\n######################')
        
    final_results_df = pd.concat(final_results, ignore_index=True)
    final_results_df['score_rules'] = [int(true == pred) for (true, pred) in zip(final_results_df['true_label'], final_results_df['pred_rule_based'])]

    final_results_df.to_csv(f"{path}{directory_names[1]}/full classification results.csv", index=False)
    print(f'After {num_reps} replications\n  Rules-\n  Accuracy: {np.round(np.mean(all_reps_accuracies_rules), 3)}\nMean Absolute Error:    {np.round(np.mean(all_reps_maes_rules), 3)}\nStandard deviation: {np.round(np.std(all_reps_accuracies_rules), 3)}')   
    print(f'\nFew Shot-\n  Accuracy: {np.round(np.mean(all_reps_accuracies_few), 3)}\nMean Absolute Error:    {np.round(np.mean(all_reps_maes_few), 3)}\nStandard deviation: {np.round(np.std(all_reps_accuracies_few), 3)}')
    print(f'\nZero shot-\n  Accuracy: {np.round(np.mean(all_reps_accuracies_zero), 3)}\nMean Absolute Error:    {np.round(np.mean(all_reps_maes_zero), 3)}\nStandard deviation: {np.round(np.std(all_reps_accuracies_zero), 3)}')


    # Prepare confusion matrices for rules, few and zero methods
    plot_confusion_matrix(y_true = final_results_df['true_label'], y_pred = final_results_df['pred_rule_based'], labels = ['High', 'Moderate', 'Low'], title = 'Rule based Model Results', save_path=f'{path}{directory_names[1]}/Rule-based Model Results')
    plt.clf()  # clear current figure
    plot_confusion_matrix(y_true = final_results_df['true_label'], y_pred = final_results_df['few_shot_pred'],   labels = ['High', 'Moderate', 'Low'], title = 'Few Shot Model Results', save_path=f'{path}{directory_names[1]}/Few Shot Model Results')
    plt.clf()  # clear current figure
    plot_confusion_matrix(y_true = final_results_df['true_label'], y_pred = final_results_df['zero_shot_pred'],  labels = ['High', 'Moderate', 'Low'], title = 'Zero Shot Model Results', save_path=f'{path}{directory_names[1]}/Zero Shot Model Results')
        
    # Analyze rules Matching, Accuracy, Mean Absolute Error


import traceback

if __name__ == "__main__":
    
    start = datetime.now()
    start_txt = start.strftime("%Y-%m-%d_%H-%M-%S")
    print(start_txt, '\n')
    
    # -------Running Main Script--------
    try:
        main_recursive(personality_trait='e', num_reps=1, start_txt=start_txt)
        
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
        with open("C:/Users/User/OneDrive/Desktop/Masters/thesis/code 3 classes/error_log.txt", "w") as f:
            f.write(traceback.format_exc())  # main_recursive()
    
        
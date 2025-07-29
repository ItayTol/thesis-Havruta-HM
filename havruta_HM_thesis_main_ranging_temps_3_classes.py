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
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.utils import shuffle
from datetime import datetime
from havruta_HM_thesis_prompts_3_classes import load_trait
from havruta_HM_thesis_load_rules_3_classes import load_rules_with_gpt
from havruta_HM_thesis_classification import classifier_zero_and_few_shot, recursive_rule_filtering, classifier_using_rules
import traceback

openai.api_key = os.getenv('OPENAI_API_KEY')
pd.set_option('display.max_colwidth', None)  # Show full text in cells

# import matplotlib.pyplot as plt
# import seaborn as sns

# -----------------------------------------------------------------------

# Load dataset (social media posts + true labels)
def load_posts_and_trait_true_label(personality_trait: str):
    df = pd.read_csv("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/true_labels_3_classes.csv", encoding='cp1252')
    return df[['p', 'post1', 'post2', f'{personality_trait}']]

# import numpy as np

def custom_fixed_slices(df, train_size=15, val_size=15, test_size=30):
    n_samples = len(df)
    assert train_size + val_size + test_size == n_samples
    assert n_samples % test_size == 0

    splits = []
    n_folds = n_samples // test_size

    for i in range(n_folds):
        test_start = i * test_size        #0,  15, 30, 45
        test_end = test_start + test_size #15, 30, 45, 60
        test_data = df.iloc[test_start:test_end] #0 to 14   # slice rows correctly

        remaining_indices = list(range(0, test_start)) + list(range(test_end, n_samples))
        remaining_data = df.iloc[remaining_indices]  # use .iloc here!

        val_data = remaining_data.iloc[:val_size]
        train_data = remaining_data.iloc[val_size:val_size + train_size]

        splits.append({
            "fold": i + 1,
            "train": train_data,
            "val": val_data,
            "test": test_data
        })

    return splits

def time_stamp(text: str):
        now = datetime.now()
        now_txt = now.strftime("%Y-%m-%d_%H-%M-%S")
        print(text, now_txt)
        return now_txt
        
def main(personality_trait: str, num_reps: int):
    participants = load_posts_and_trait_true_label(personality_trait)
    dict_class_to_code = {'Unknown': -10, 'Low': 1, 'Moderate': 2, 'High': 3}
    
    # Specify the directory name
    start_txt = time_stamp('START')
    path_trait = f'C:/Users/User/OneDrive/Desktop/Masters/thesis/code 3 classes/{load_trait(personality_trait)}/'
    run_folder = f'run_{start_txt}/'
    os.mkdir(''.join(path_trait + run_folder))
    path = path_trait + run_folder
    directory_names = ["loaded_rules", "classification_results", "validation set", "Top Rules"] 
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
        participants = shuffle(participants).reset_index(drop=True)
        
        splits = custom_fixed_slices(participants)
  
        for fold in splits:
            print(f"      -----Fold {fold['fold']}   (rep {rep}) ------")
            train = fold['train']
            val = fold['val']
            test = fold['test']
            
            # print('train\n', train[['p']], '\n\n', 'val\n', val['p'], '\n\n' ,'test\n', test['p'])
            train_participants = train['p'].to_list()

            n = len(train_participants)
            chunk_size = n // 3  # Integer division
            
            # Create the three sub-lists using slicing
            list1 = train_participants[0:chunk_size]
            list2 = train_participants[chunk_size:2 * chunk_size]
            list3 = train_participants[2 * chunk_size:] # The rest of the list
            
            rules = []
            print('Loading Rules')
            rules_sets = []
            rules_sets.append(load_rules_with_gpt(personality_trait, list1))
            # rules_sets.append(load_rules_with_gpt(personality_trait, list2))
            # rules_sets.append(load_rules_with_gpt(personality_trait, list3))
            for rules_set in rules_sets:
                for rule in rules_set:
                    rules.append(rule)
                    
            print('Rules Loaded Successfully')
            # Concatenate folds and save per rep        
            with open(f"{path}{directory_names[0]}/Loaded Rules rep {rep} fold {fold['fold']}.txt", 'w') as f:
                for item in rules:
                    f.write(str(item) + '\n')    
            print('Exporting Rules Is Completed')

            # Build few-shot prompt using training data
            extra_prompt_few_shot = ""
            for i, row in train.sample(5).iterrows():
                extra_prompt_few_shot += (
                    f"Example {i+1}\nInput:\n1. {row['post1']}\n2. {row['post2']}\n"
                    f"Output: {row[personality_trait]}\n\n")
            print('extra_prompt_few_shot loaded successfully')
# -------------------------------------------------------------------------------------------------
            # Get most helpful rules using validation data
            # print('min_matches:', len(val['p'])/2)
            # result_val_rules = recursive_rule_filtering(personality_trait, val, rules, 
            #                                             rep, fold, len(val['p'])/2, 0.67,
            #                                             4, path, directory_names)
            
            result_val_rules = classifier_using_rules(personality_trait, val, rules)
           
            result_val_rules.to_csv(f"{path}{directory_names[2]}/classification validation set results rep {rep} fold {fold['fold']}.csv", index=False)
            
            return None
            
# -------------------------------------------------------------------------------------------------
            # Evaluate top rules on test data             
            result_test_rules = classifier_using_rules(personality_trait, test, top_rules)
            result_test_rules['fold'] = fold
            rep_results_test_rules.append(result_test_rules) 

            
            # Evaluate few and zero shot on test data             
            result_test_zero_and_few_shot = classifier_zero_and_few_shot(personality_trait, test, extra_prompt_few_shot, rep, fold) 
            rep_results_test_zero_few_shot.append(result_test_zero_and_few_shot)
# -------------------------------------------------------------------------------------------------

                
        rep_test_rules = pd.concat(rep_results_test_rules, ignore_index=True)
        rep_test_zero_few_shot = pd.concat(rep_results_test_zero_few_shot, ignore_index=True)
        
        rep_test_rules["matched_rules"] = rep_test_rules["matched_rules"].apply(lambda r: "; ".join(r) if isinstance(r, list) else str(r))
        # rep_test_rules["matched_behaviours"] = rep_test_rules["matched_behaviours"].apply(lambda r: "; ".join(r) if isinstance(r, list) else str(r))
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


if __name__ == "__main__":

    
    try:
        start = datetime.now()
        main(personality_trait = 'e', num_reps = 1)
        
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
    end = datetime.now()
    end_txt = end.strftime("%Y-%m-%d_%H-%M-%S")
    print('END', end_txt)
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:54:33 2025

@author: User
"""

# If the accuracy is up after adding a rule, than the rule is important. if else, the rule doesn't matter.
def stepwise_selection_rule(personality_trait: str, train_X, train_y, rules: list):
    best_accuracy = validation_zero_shot(personality_trait, train_X, train_y) #Zero Shot 
    print('Baseline - No rules:', best_accuracy)
    best_rules = []

    for rule in rules:
        temp_rules = best_rules + [rule]
        new_accuracy, predictions = validation_rules(personality_trait, train_X, train_y, temp_rules)

        if new_accuracy > best_accuracy:  # Add the rule if old accuracy is bigger than new accuracy
            best_rules.append(rule)
            best_accuracy = new_accuracy
            # print(f"SELECTED - Rule '{rule}', accuracy is {new_accuracy}")
        # else:
            # print(f"IGNORED - Rule '{rule}', accuracy is {new_accuracy}")    
    print(f'Using all selected rules accuracy is: {best_accuracy}')
    return best_accuracy, best_rules

# if the accuracy drops after removing a rule, than the rule is important. if else, the rule doesn't matter
def stepwise_elimination_rule(personality_trait: str, train_X, train_y, rules: list):
    best_accuracy, predictions = validation_rules(personality_trait, train_X, train_y, rules) #accuracy with all rules
    print('Baseline - All rules:', best_accuracy)
    best_rules = rules.copy()
    
    for rule in rules:
        temp_rules = best_rules.copy()
        temp_rules.remove(rule)
        
        new_accuracy, predictions = validation_rules(personality_trait, train_X, train_y, temp_rules)

        if new_accuracy >= best_accuracy :  # Remove this rule if accuracy without this rule is bigger than accuracy with all rules 
            best_rules = temp_rules
            best_accuracy = new_accuracy
            # print(f"REMAINED - Rule '{rule}', accuracy is {new_accuracy}")
        # else: 
            # print(f"REMOVED - Rule '{rule}', accuracy is {new_accuracy}")
    print(f'Using all remained rules accuracy is: {best_accuracy}')
    return best_accuracy, best_rules

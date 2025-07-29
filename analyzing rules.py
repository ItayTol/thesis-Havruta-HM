# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 18:47:44 2025

@author: User
"""
import pandas as pd
from collections import Counter
df = pd.read_csv("C:/Users/User/OneDrive/Desktop/Masters/thesis/code 3 classes/extraversion/run_2025-07-04_12-50-28/classification_results/classification results combined.csv")

helpful_rules = df['helpful_rules'].dropna()
rules_flattened = []
for many_rules in helpful_rules:
    rules_list = many_rules.split(';')
    for rule in rules_list:
        rules_flattened.append(rule)

counts = Counter(rules_flattened)

print(f"counts: {(list(counts.most_common(15)))}")

# from ast import literal_eval

# li_rule_to_category = []
# with open("C:/Users/User/OneDrive/Desktop/Masters/thesis/code 3 classes/extraversion/run_2025-07-04_12-50-28/rule_to_category/iLoveMerge.txt") as f:
#     for line in f:
#         if line == '\n':
#             continue
#         else:
#             li_rule_to_category.append(literal_eval(line))

# rules = []
# categories = []
# for i, j in li:
#     rules.append(i)
#     categories.append(j)
# rule_to_category = pd.DataFrame({"rule" : rules, "category" : categories})

# for rule in list(counts.most_common(20)):
#     if rule in 


    
    
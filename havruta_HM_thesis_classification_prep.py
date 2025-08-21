# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 00:01:12 2025

@author: User
"""

import json
from havruta_HM_thesis_gpt_response_3_classes import standard_gpt_response
from havruta_HM_thesis_prompts_3_classes import prompt_for_shot_classification, prompt_classify_with_rules

# Get a singular prediction - zero shot
def evaluate_zero_shot(personality_trait, post1, post2):    
    # Make prompt for gpt
    system_content, user_content = prompt_for_shot_classification(personality_trait, post1, post2)
    # Get Prediction to text
    response = standard_gpt_response(system_content, user_content)
    try:
        lines = response.strip().splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
           lines = lines[:-1]
        return json.loads("\n".join(lines))
      
    except:
        return "error", "Invalid JSON format."
    return response

# Get a singular prediction - few shot
def evaluate_few_shot(personality_trait, post1, post2, few_shot):    
    # Make prompt for gpt
    partial_system_content, user_content = prompt_for_shot_classification(personality_trait, post1, post2)
    system_content = f'''{partial_system_content}
    In your response follow these examples:
    {few_shot}'''
    # Get Prediction to text
    response = standard_gpt_response(system_content, user_content)
    try:
        lines = response.strip().splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
           lines = lines[:-1]
        return json.loads("\n".join(lines))      
    except:
        return "error", "Invalid JSON format."
    return response

# Get a singular prediction - few shot
def evaluate_academic_rules_shot(personality_trait, post1, post2, academic_rules):    
    # Make prompt for gpt
    partial_system_content, user_content = prompt_for_shot_classification(personality_trait, post1, post2)
    system_content = f'''{partial_system_content}
    In your response follow these examples:
    {academic_rules}'''
    # Get Prediction to text
    response = standard_gpt_response(system_content, user_content)
    try:
        lines = response.strip().splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return json.loads("\n".join(lines))    
    except:
        return "error", "Invalid JSON format."
    return response


def rank_rules_and_classify(personality_trait, post1, post2, rules: list):
    system_content, user_content = prompt_classify_with_rules(personality_trait, post1, post2, rules)
    response = standard_gpt_response(system_content, user_content)
    try:
        lines = response.strip().splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
           lines = lines[:-1]
        return json.loads("\n".join(lines))
      
    except:
        print(response)
        return "error", "Invalid JSON format."
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 00:01:12 2025

@author: User
"""

import json
import hashlib, time
from collections import defaultdict, Counter
from havruta_HM_thesis_gpt_response_3_classes import standard_gpt_response
from havruta_HM_thesis_prompts_3_classes import prompt_for_shot_classification, prompt_classify_with_rules, load_trait


def llm_call(system_prompt:str, user_prompt:str) -> dict:
    raw = standard_gpt_response(system_prompt, user_prompt)
    # Strip code fences if the model returns them
    lines = [ln for ln in raw.strip().splitlines()]
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    
    except Exception as e:
        # Surface the bad output in the exception to debug prompt errors quickly
        raise ValueError(f"Model did not return valid JSON. \n{text}") from e
    return parsed

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


# Get a singular prediction - rules
def rank_rules_and_classify(personality_trait, post1, post2, rules):    
    # Make prompt for gpt
    partial_system_content, user_content = prompt_classify_with_rules(personality_trait, post1, post2, rules)
    system_content = f'''{partial_system_content}
    In your response follow these rules:
    {rules}'''
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

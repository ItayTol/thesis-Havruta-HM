# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:57:14 2025

@author: User
"""

import pandas as pd
import json

pd.set_option('display.max_colwidth', None)  # Show full text in cells
pd.set_option('display.max_columns', None)

from havruta_HM_thesis_prompts_3_classes import prompt_create_rules_low_level, prompt_create_rules_high_level
from havruta_HM_thesis_gpt_response_3_classes import standard_gpt_response, warm_gpt_response, hot_gpt_response, cold_gpt_response

def load_rules_with_human(personality_trait: str, start_p: int, end_p: int, correct = None):
    # read sheet of personality trait
    df_self_made_rules = pd.read_excel("C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/rules_made_by_me.xlsx", personality_trait)
    # replace ' ' with '_' in column names
    df_self_made_rules.columns = [column.replace(" ", "_") for column in df_self_made_rules.columns]
    # change occurence datatype to str
    df_self_made_rules = df_self_made_rules.astype({"Occurences": str})
    # filter only true occurences, not rules that are included in other rules
    df_self_made_rules = df_self_made_rules.query('Occurences.notna() and Occurences.str.isdigit()', engine="python")
    # Slice from start to end participant
    df_self_made_rules = df_self_made_rules.query(f'p >= {start_p} & p <= {end_p}')
    # sort rules by occurence
    df_self_made_rules = df_self_made_rules.sort_values(by = 'Occurences', ascending = False)
    if correct:
        df_self_made_rules = df_self_made_rules.query('True_Rating == Psych_Rating')['Rule_Content_For_GPT'] 
    else: 
        df_self_made_rules = df_self_made_rules['Rule_Content_For_GPT'] 
    return df_self_made_rules

def get_assessments(df, psych_rating):
    output = ""
    pair_ids = df['p'].unique()  # assumes each post pair has a unique ID
    # print(pair_ids)
    counter = 1
    for pair_id in pair_ids:
        group = df[df['p'] == pair_id]
        for _, row in group.iterrows():
            if row['psych_rating'] in [psych_rating, "Moderate"]:
                output += f"{counter}\n"
                # output += f"Post 1: {row['post1']}\n"
                # output += f"Post 2: {row['post2']}\n"
                # output += f"True rating: {row['true_rating']}\n"
                output += f"Psychologist rating: {row['psych_rating']}\n"
                output += f"Psychologist comment: {row['reason']}\n\n"
                counter += 1
    return output

def prep_rules_response_for_loading(res):
    lines = res.strip().splitlines()   
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
       lines = lines[:-1]
    return json.loads("\n".join(lines))

def load_rules_with_gpt(personality_trait: str, participants: list):
    df = pd.read_excel('C:/Users/User/OneDrive/Desktop/Masters/thesis/DATA/rules_made_by_gpt.xlsx', personality_trait)  
    df = df[df['p'].isin(participants)]    
    df = df.reset_index(drop = True)  
    df.columns = [column.replace(" ", "_") for column in df.columns]

    low_assessments = get_assessments(df, 'Low')
    high_assessments = get_assessments(df, 'High')

    # Get detail of assessments
    # size_low_assessments = len(df[df['psych_rating'] == 'Low'])
    # size_high_assessments = len(df[df['psych_rating'] == 'High'])

    # print('Number of low assessments:', size_low_assessments)
    # print(df[df['psych_rating'] == 'Low'][['code', 'score']])
    # print('\n\n')
    # print('Number of high assessments:', size_high_assessments)
    # print(df[df['psych_rating'] == 'High'][['code', 'score']])
    # print('\n\n')
    
    # Generate rules
    
    standard_res = []
    # warm_res, hot_res, cold_res = [], [], []
    # overall_res = []
    
    low_level_system_content, low_level_user_content = prompt_create_rules_low_level(personality_trait, low_assessments)
    high_level_system_content, high_level_user_content = prompt_create_rules_high_level(personality_trait, high_assessments)
    

    try:
        res = standard_gpt_response(low_level_system_content, low_level_user_content)
        standard_low_level_response_generated = prep_rules_response_for_loading(res)
    except:
        print('problem in standard_low_level_response')
        print(res)
        return None
    standard_res += standard_low_level_response_generated
    
    # High level, Standard temp
    try:
        res = standard_gpt_response(high_level_system_content, high_level_user_content)
        standard_high_level_response_generated = prep_rules_response_for_loading(res)
    except:
        print('problem in standard_high_level_response')
        print(res)
        return None
    standard_res += standard_high_level_response_generated
    
    return standard_res
x='''
    if temp == 'cold':
        # Low level, Cold temp
        try:
            res = cold_gpt_response(low_level_system_content, low_level_user_content)
            cold_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in cold_low_level_response')
            print(res)
            return None
        cold_res += cold_low_level_response_generated
        
        # High level, Cold temp
        try:
            res = cold_gpt_response(high_level_system_content, high_level_user_content)
            cold_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in cold_high_level_response')
            print(res)
            return None
        cold_res += cold_high_level_response_generated
        
        return cold_res
    
    
    if temp == 'warm':
    
        # Low level, Warm temp
        try:
            res = warm_gpt_response(low_level_system_content, low_level_user_content)
            warm_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in warm_low_level_response')
            print(res)
            return None
        warm_res += warm_low_level_response_generated
        
        # High level, Warm temp
        try:    
            res = warm_gpt_response(high_level_system_content, high_level_user_content)
            warm_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in warm_high_level_response')
            print(res)
            return None
        warm_res += warm_high_level_response_generated
    
        return warm_res


    if temp == 'hot':

        # Low level, Hot temp
        try:
            res = hot_gpt_response(low_level_system_content, low_level_user_content)
            hot_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in hot_low_level_response')
            print(res)
            return None
        hot_res += hot_low_level_response_generated
    
        # High level, Hot temp
        try:
            res = hot_gpt_response(high_level_system_content, high_level_user_content)
            hot_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in hot_high_level_response')
            print(res)
            return None
        hot_res += hot_high_level_response_generated
        
        return hot_res

    
    if temp == 'standard and warm':
        # Low level, Standard temp
        try:
            res = standard_gpt_response(low_level_system_content, low_level_user_content)
            standard_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in standard_low_level_response')
            print(res)
            return None
        standard_res += standard_low_level_response_generated
        
        # High level, Standard temp
        try:
            res = standard_gpt_response(high_level_system_content, high_level_user_content)
            standard_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in standard_high_level_response')
            print(res)
            return None
        standard_res += standard_high_level_response_generated
        
        # Low level, Warm temp
        try:
            res = warm_gpt_response(low_level_system_content, low_level_user_content)
            warm_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in warm_low_level_response')
            print(res)
            return None
        warm_res += warm_low_level_response_generated
        
        # High level, Warm temp
        try:    
            res = warm_gpt_response(high_level_system_content, high_level_user_content)
            warm_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in warm_high_level_response')
            print(res)
            return None
        warm_res += warm_high_level_response_generated
    
        return standard_res + warm_res

            
    if temp == 'standard and hot':
       try:
           res = standard_gpt_response(low_level_system_content, low_level_user_content)
           standard_low_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in standard_low_level_response')
           print(res)
           return None
       standard_res += standard_low_level_response_generated
       
       # High level, Standard temp
       try:
           res = standard_gpt_response(high_level_system_content, high_level_user_content)
           standard_high_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in standard_high_level_response')
           print(res)
           return None
       standard_res += standard_high_level_response_generated


       # Low level, Hot temp
       try:
           res = hot_gpt_response(low_level_system_content, low_level_user_content)
           hot_low_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in hot_low_level_response')
           print(res)
           return None
       hot_res += hot_low_level_response_generated
   
       # High level, Hot temp
       try:
           res = hot_gpt_response(high_level_system_content, high_level_user_content)
           hot_high_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in hot_high_level_response')
           print(res)
           return None
       hot_res += hot_high_level_response_generated
       
       return hot_res

           
    if temp == 'warm and hot':
       try:
           res = warm_gpt_response(low_level_system_content, low_level_user_content)
           warm_low_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in warm_low_level_response')
           print(res)
           return None
       warm_res += warm_low_level_response_generated
       
       # High level, Warm temp
       try:
           res = warm_gpt_response(high_level_system_content, high_level_user_content)
           warm_high_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in warm_high_level_response')
           print(res)
           return None
       warm_res += warm_high_level_response_generated


       # Low level, Hot temp
       try:
           res = hot_gpt_response(low_level_system_content, low_level_user_content)
           hot_low_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in hot_low_level_response')
           print(res)
           return None
       hot_res += hot_low_level_response_generated
   
       # High level, Hot temp
       try:
           res = hot_gpt_response(high_level_system_content, high_level_user_content)
           hot_high_level_response_generated = prep_rules_response_for_loading(res)
       except:
           print('problem in hot_high_level_response')
           print(res)
           return None
       hot_res += hot_high_level_response_generated
       
       return hot_res

 
    if temp == 'all':
        
        # Low level, Standard temp
        try:
            res = standard_gpt_response(low_level_system_content, low_level_user_content)
            standard_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in standard_low_level_response')
            print(res)
            return None
        overall_res += standard_low_level_response_generated
        
        # High level, Standard temp
        try:
            res = standard_gpt_response(high_level_system_content, high_level_user_content)
            standard_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in standard_high_level_response')
            print(res)
            return None
        overall_res += standard_high_level_response_generated
        
        
        # Low level, Warm temp
        try:
            res = warm_gpt_response(low_level_system_content, low_level_user_content)
            warm_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in warm_low_level_response')
            print(res)
            return None
        overall_res += warm_low_level_response_generated
        
        # High level, Warm temp
        try:    
            res = warm_gpt_response(high_level_system_content, high_level_user_content)
            warm_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in warm_high_level_response')
            print(res)
            return None
        overall_res += warm_high_level_response_generated
        
        # Low level, Hot temp
        try:
            res = hot_gpt_response(low_level_system_content, low_level_user_content)
            hot_low_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in hot_low_level_response')
            print(res)
            return None
        overall_res += hot_low_level_response_generated
        
        # High level, Hot temp
        try:
            res = hot_gpt_response(high_level_system_content, high_level_user_content)
            hot_high_level_response_generated = prep_rules_response_for_loading(res)
        except:
            print('problem in hot_high_level_response')
            print(res)
            return None
        overall_res += hot_high_level_response_generated
        
        return overall_res
'''
 
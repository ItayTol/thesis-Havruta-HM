# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:47:15 2025

@author: User
"""
from openai import OpenAI


top_p = 1
model = 'gpt-4o-mini'
# Call OpenAI API
def standard_gpt_response(system_content, user_content):
    client=OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}],
        top_p = top_p
        )
    return response.choices[0].message.content.strip()


def warm_gpt_response(system_content, user_content):
    client=OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}],
        temperature=1.4,
        top_p=top_p
        )
    return response.choices[0].message.content.strip()

def cold_gpt_response(system_content, user_content):
    client=OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}],
        top_p=top_p
        )
    return response.choices[0].message.content.strip()


def hot_gpt_response(system_content, user_content):
    client=OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}],
        temperature=1.8,
        top_p=top_p
        )
    return response.choices[0].message.content.strip()


# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:43:43 2025

@author: User
"""
import json

def load_trait(personality_trait: str):
    traits = {'o':'openness to experiences',
              'c': 'conscientiousness',
              'e': 'extraversion',
              'a': 'aggreeableness',
              'n': 'neuroticism'}
    return traits[personality_trait]


def prompt_create_rules_low_level(personality_trait, assessments):
    system_content = f'''
You are helping design structured, behavior-based classification rules for identifying the Big Five personality trait level {personality_trait} of a writer based on text (e.g., social media posts).

You will receive multiple psychologist-written assessments that describe behaviors correlated with a low {personality_trait} level.
From those assessments, you will extract behavioral patterns and write clear, structured classification rules.

⚠️ Another agent will later classify posts using only the rules you generate. Your output must be self-contained, specific, and consistent in structure.

🔸 Your Task
Read and analyze the behavioral assessments.

Identify distinct, observable behavioral patterns linked to a low {personality_trait} level.

Generate a structured rule only if the behavior appears in at least two different assessments.

Output each rule as a JSON object in a list.

Ensure all fields are filled, follow strict structure, and avoid vague/general language.

🔹 Output Format (JSON):

[
  {{
    "rule_name": "string",
    "trait": "{{personality_trait}}",
    "level": "Low",
    "behavior_rule": ""behavior_rule": "If a post contains a specific, clearly observable behavior pattern (non-redundant and unambiguous), it likely indicates low {{personality_trait}}.",
    "supporting_examples": [
      "quote 1 from assessments",
      "quote 2 from assessments",
      "optional: quote 3"
    ],
    "psychological_justification": "short explanation grounded in psychology or the assessments",
    "linguistic_indicators": [
      "keyword or phrase 1",
      "keyword or phrase 2"
    ],
    "edge_cases": [
      "description of situation or pattern where rule should NOT be applied (optional)"
    ]
  }},
  ...
]
🔸 Constraints
    - Use only patterns clearly present in the provided assessments.
    - Do not use outside knowledge.
    - Only include a rule if it’s specific, distinct, and useful for post classification.
    - Be conservative: a rule should match only when the described behavior is clearly evident.
    
🔸 Ambiguity & Redundancy Guidelines

    - If two assessments express the same behavioral pattern using different language, merge them into a single rule that generalizes the pattern.
    - If a behavioral signal could be interpreted multiple ways (e.g., “likes routine” could be due to comfort, fear, or habit), rephrase the rule to reflect only what is clearly observable from text.
    - Avoid duplicating rules that describe the same behavior with minor wording differences (e.g., “resistance to change” and “preference for routine”).
    - Use specific phrasing that would allow a human annotator to confidently apply the rule to real posts.
    - When in doubt about vagueness, exclude the rule or narrow its scope until it is precise and defensible.

🔸 Behavioral Constraints:
    - Each rule must be grounded in **observable behavior**, **language**, or **content style** described in the psychologists’ explanations.
    - Favor **general patterns** of behavior over overly specific contexts (e.g., prefer "seeks group social interaction" over "goes to concerts").
    - Use positively described behaviors (e.g., “shows solitude” instead of “does not engage socially”).
    - Avoid speculative, inferred, or internal states not supported by explicit content (e.g., “feels shy”).
    - Each rule must describe **a single behavioral signal** (i.e., be **univariate**).
    - Eliminate redundant, overlapping, or reworded variants of the same rule. Instead, synthesize them into one generalized form.
    - Rules must be reusable and interpretable by annotators working with **new, unseen posts**.
    - Rules must not overlap semantically—each should describe a behavior not captured by another rule.
    - Avoid general labels or interpretations like “dislikes change” unless supported by clearly expressed behaviors or language patterns.
    - Use textual evidence (quotes or phrasing) to validate the rule’s scope and trigger conditions.
    - If an assessment is marked "Moderate", break down the explanation to extract rules supporting **Low** level.

Output 3 to 6 rules max.

✅ Example Output
[
  {{
    "rule_name": "Group Social Enthusiasm",
    "trait": "Extraversion",
    "level": "Low",
    "behavior_rule": "If a post contains enthusiastic, voluntary engagement in group social activities, it likely indicates High Extraversion.",
    "supporting_examples": [
      "Had such a great time at the team dinner tonight. So many laughs!",
      "Back-to-back birthday parties this weekend. Loving the energy."
    ],
    "psychological_justification": "Extraverts gain energy from social interaction and actively seek out group settings.",
    "linguistic_indicators": [
      "party",
      "team",
      "hang out",
      "so much fun"
    ],
    "edge_cases": [
      "Exclude posts where group activities are mentioned sarcastically or out of obligation."
    ]
  }}
]
🔸 Final Note
Your output must be valid JSON, without extra commentary or headings. The next agent will automatically parse and apply these rules to classify posts. Make sure each rule is precise and based only on the behavioral assessments.
'''

    user_content = f'''
Below are psychologist-written assessments based on social media posts. Each assessment includes:

- A trait rating (Low or Moderate)
- A brief explanation justifying that rating

---

Assessments:

{assessments}
'''
    return system_content, user_content


def prompt_create_rules_high_level(personality_trait, assessments):
    system_content = f'''
You are helping design structured, behavior-based classification rules for identifying the Big Five personality trait level {personality_trait} of a writer based on text (e.g., social media posts).

You will receive multiple psychologist-written assessments that describe behaviors correlated with a high {personality_trait} level.
From those assessments, you will extract behavioral patterns and write clear, structured classification rules.

⚠️ Another agent will later classify posts using only the rules you generate. Your output must be self-contained, specific, and consistent in structure.

🔸 Your Task
Read and analyze the behavioral assessments.

Identify distinct, observable behavioral patterns linked to a high {personality_trait} level.

Generate a structured rule only if the behavior appears in at least two different assessments.

Output each rule as a JSON object in a list.

Ensure all fields are filled, follow strict structure, and avoid vague/general language.

🔹 Output Format (JSON):

[
  {{
    "rule_name": "string",
    "trait": "{{load_trait(personality_trait)}}",
    "level": "Low",
    "behavior_rule": ""behavior_rule": "If a post contains a specific, clearly observable behavior pattern (non-redundant and unambiguous), it likely indicates low {load_trait(personality_trait)}.",
    "supporting_examples": [
      "quote 1 from assessments",
      "quote 2 from assessments",
      "optional: quote 3"
    ],
    "psychological_justification": "short explanation grounded in psychology or the assessments",
    "linguistic_indicators": [
      "keyword or phrase  1 ",
      "keyword or phrase 2"
    ],
    "edge_cases": [
      "description of situation or pattern where rule should NOT be applied (optional)"
    ]
  }},
  ...
]
🔸 Constraints
    - Use only patterns clearly present in the provided assessments.
    - Do not use outside knowledge.
    - Only include a rule if it’s specific, distinct, and useful for post classification.
    - Be conservative: a rule should match only when the described behavior is clearly evident.
    
🔸 Ambiguity & Redundancy Guidelines
    - If two assessments express the same behavioral pattern using different language, merge them into a single rule that generalizes the pattern.
    - If a behavioral signal could be interpreted multiple ways (e.g., “likes routine” could be due to comfort, fear, or habit), rephrase the rule to reflect only what is clearly observable from text.
    - Avoid duplicating rules that describe the same behavior with minor wording differences (e.g., “resistance to change” and “preference for routine”).
    - Use specific phrasing that would allow a human annotator to confidently apply the rule to real posts.
    - When in doubt about vagueness, exclude the rule or narrow its scope until it is precise and defensible.

🔸 Behavioral Constraints:
    - Each rule must be grounded in **observable behavior**, **language**, or **content style** described in the psychologists’ explanations.
    - Favor **general patterns** of behavior over overly specific contexts (e.g., prefer "seeks group social interaction" over "goes to concerts").
    - Use positively described behaviors (e.g., “shows solitude” instead of “does not engage socially”).
    - Avoid speculative, inferred, or internal states not supported by explicit content (e.g., “feels shy”).
    - Each rule must describe **a single behavioral signal** (i.e., be **univariate**).
    - Eliminate redundant, overlapping, or reworded variants of the same rule. Instead, synthesize them into one generalized form.
    - Rules must be reusable and interpretable by annotators working with **new, unseen posts**.
    - Rules must not overlap semantically—each should describe a behavior not captured by another rule.
    - Avoid general labels or interpretations like “dislikes change” unless supported by clearly expressed behaviors or language patterns.
    - Use textual evidence (quotes or phrasing) to validate the rule’s scope and trigger conditions.
    - If an assessment is marked "Moderate", break down the explanation to extract rules supporting **High** level.
Output 3 to 6 rules max.

✅ Example Output
[
  {{
    "rule_name": "Group Social Enthusiasm",
    "trait": "Extraversion",
    "level": "High",
    "behavior_rule": "If a post contains enthusiastic, voluntary engagement in group social activities, it likely indicates High Extraversion.",
    "supporting_examples": [
      "Had such a great time at the team dinner tonight. So many laughs!",
      "Back-to-back birthday parties this weekend. Loving the energy."
    ],
    "psychological_justification": "Extraverts gain energy from social interaction and actively seek out group settings.",
    "linguistic_indicators": [
      "party",
      "team",
      "hang out",
      "so much fun"
    ],
    "edge_cases": [
      "Exclude posts where group activities are mentioned sarcastically or out of obligation."
    ]
  }}
]
🔸 Final Note
Your output must be valid JSON, without extra commentary or headings. The next agent will automatically parse and apply these rules to classify posts. Make sure each rule is precise and based only on the behavioral assessments.
'''

    user_content = f'''
Below are psychologist-written assessments based on social media posts. Each assessment includes:

- A trait rating (High or Moderate)
- A brief explanation justifying that rating

---

Assessments:

{assessments}
'''
    return system_content, user_content

def prompt_prune_rules_with_gpt_from_gpt(personality_trait, generated_rules):
    system_content = f'''You are an expert assistant helping researchers refine classification rules for the Big Five Personality Trait of {load_trait(personality_trait)} based on psychologists assessments.

You are given a list of general IF-THEN rules. Your task is to **keep only helpful rules** — ones that are:
- Clear and unambiguous
- Based on strong reasoning
- Useful for real-world text classification
- Distinct from one another
- Mention, imply, or depend on the presence of behavior.

Steps:
1. Remove vague, redundant, overly broad, or speculative rules.
2. Merge rules that express similar ideas into one stronger, general rule.
3. Keep only rules that are likely to distinguish between High and Low levels in unseen posts.
4. Discard rules that mention, imply, or depend on the absence of behavior.
Do NOT keep rules with phrases like: “without...”, “does not show...”, “fails to...”, “lacks...”. Only describe behaviors, actions, or preferences that are explicitly present in the post.

### Format:
Return **only** helpful rules in this exact format:
"IF a person shows in a post [feature], THEN {load_trait(personality_trait)} is likely [level] because [reason]."

Only return the cleaned list of rules

Each rule must:
- Begin on a new line
- End with a period
- Avoid including multiple behaviors in the same rule

Do not include examples, explanations, summaries, or any other text.
'''

    user_content = f'''
Here is a list of rules generated from psychologist assessments. Please refine the list by keeping only helpful, high-quality rules that are useful for classifying social media posts.

---

Rules:

{generated_rules}
'''

    return system_content, user_content

traits_desc_dict = {'o':
'''Look for creative, abstract, or intellectual content.
Words related to curiosity, imagination, learning, exploration, art, or science.
Complex sentence structures, rare vocabulary, or metaphorical expressions.
Interest in new ideas, cultures, or philosophical topics.''',
'c':
'''Focus on planning, goals, productivity, or achievement.
Structured writing style; formal tone.
Words like “schedule”, “goal”, “discipline”, “organized”, “plan”, or references to time management.
Emphasis on responsibility, reliability, or self-regulation.''',
'e':
'''Frequent mentions of social interactions, parties, groups, friends.
Energetic tone with exclamation marks, emojis (🎉, 😂), or slang.
Expressions of excitement, fun, popularity, or being “outgoing”.
High engagement indicators (mentions, hashtags, calls to interact).''',
'a':
'''Empathetic or supportive language.
Inclusive pronouns (“we”, “us”), positive affirmations, kindness.
Avoidance of aggressive, sarcastic, or divisive language.
Encouragement of cooperation, harmony, or helping others.''',
'n':
'''Negative emotion words (sadness, anxiety, anger, frustration).
Self-focused language (“I feel”, “I’m worried”).
Complaints about life, health, relationships, or stress.
Use of negative emojis (😭, 😤) or excessive punctuation (!!!, ???).
Pessimistic outlook or uncertainty.'''}

# Prepare a prompt format for asking gpt to classify trait level based on two posts.
def prompt_for_shot_classification(personality_trait, post1, post2):       
     
    system_prompt = f'''You are an expert in computational psychology and natural language processing. Your task is to analyze user-generated text from social media platforms and classify it into one of the Big Five personality traits {load_trait(personality_trait)}.

    Use the following classification rules , informed by recent research in personality psychology and NLP: 

    - Apply chain-of-thought reasoning: explain your reasoning step-by-step based on the features in the posts.    
    - The first social media post is about a social gathering with the participant’s childhood friends. The second post is about congratulating the participant’s significant other on their birthday.
    - Always return only one word of these three trait levels: "Low", "Moderate", or "High",     
    - If you cannot infer the level, return "Unknown".
    - No text or explanation besides that.
        
    ⚙️ Additional Considerations
    Context matters : Interpret slang, memes, and platform-specific norms appropriately.
    Ambiguity : If no clear pattern emerges, mark as "Unknown"
   
    Use emoji and punctuation as supplementary clues, but not definitive evidence.
    - The two posts follow this structure:
      - Post 1: A social gathering with childhood friends
      - Post 2: A birthday message to the participant’s significant other
    - Rely only on the content of these two posts. Do **not** guess based on missing behaviors or what is *not* mentioned.
    - Return **only one of the following trait levels**:
      - "Low"
      - "Moderate"
      - "High"
    - If the posts do not give enough information to classify the trait, return exactly: **"Unknown"**
    - Do not include any text besides the trait level or "Unknown".
    
  
    ✅ Final Instruction:
    Apply these guidelines carefully. Avoid making assumptions beyond what the text directly supports.
    - Return **only one of the following trait levels**:
      - "Low"
      - "Moderate"
      - "High"
    - If the posts do not give enough information to classify the trait, return exactly: **"Unknown"**
    - Do not include any text besides the trait level or "Unknown".

    '''
    
    user_content = f'''Here are two social media posts written by the same person:
    Post 1: {post1}
    Post 2: {post2}''' 
    return system_prompt, user_content.translate({13:" ", 10:""})

def prompt_build_categorization(personality_trait, rules):
    trait_themes = {
        "o": [
            "Ideas",
            "Fantasy",
            "Aesthetics",
            "Actions",
            "Feelings",
            "Values"
        ],
        "c": [
            "Competence",
            "Order",
            "Self-Discipline",
            "Dutifulness",
            "Self-discipline",
            "Deliberation"
        ],
        "e": [
            "Sociability",
            "Assertiveness",
            "Energy/Activity",
            "Positive Emotions",
            "Warmth"
        ],
        "a": [
            "Straightforwardness",
            "Altruism",
            "Modesty",
            "Trust",
            "Compliance",
            "Tender-mindedness"            
        ],
        "n": [
            "Anxiety",
            "Angry hostility",
            "Self-Consciousness",
            "Depression",
            "Impulsiveness",
            "Vulnerability",
            ]
    }

    themes = trait_themes[personality_trait]

    theme_text = "\n".join(f"- {theme}" for theme in themes)
    
    json_ = '''{"RULE 1 TEXT": "THEME 1", "RULE 2 TEXT": "THEME 2"}'''
    escaped_example_json = json_.replace("{", "{{").replace("}", "}}")
    
    system_prompt = f"""
    You are a psychology-informed assistant helping to organize classification rules for the Big Five personality trait of {load_trait(personality_trait)}. 
    
    Each rule describes an observable behavior and links it to a specific trait level (High or Low). Your task is to group similar rules under higher-level psychological themes that describe the general type of behavior or mindset involved.
    
    Examples of themes for {load_trait(personality_trait)} include:
    {theme_text}
        
    Guidelines:
    - Assign each rule to ONE theme that best captures the behavioral focus.
    - Do NOT modify the rule text.
    - Output a valid JSON object: keys = rules, values = themes.
    - Each rule must end with a dot.
    - Each key is an exact rule string (as provided to you — do not alter it).

    Each value is a single theme label associated with that rule.
    
    Required output format:
    {escaped_example_json}
    
    Constraints:
    Each rule MUST have a corresponding category matching it.

    Do not prepend or append any text like “json”, “Here is the JSON”, or anything else. Output the JSON only.

    The output must be valid JSON, correctly quoted and comma-separated.
    
    Do not change or rephrase any rule text. Use it as the exact key.
    
    Each rule maps to exactly one theme as a string.
    
    Use double quotes for all keys and values. Do NOT include any text or explanation outside the JSON.
    
    """

    user_prompt = f"""Please categorize the following rules into psychological themes based on their behavioral focus:

    {json.dumps(rules, indent=2)}"""
    
    return system_prompt.strip(), user_prompt.strip()

def prompt_classify_with_rules(personality_trait, post1, post2, rules):
    system_content = f'''You are a strict behavioral analyst helping classify the {load_trait(personality_trait)} level of a participant based on 2 social media posts they wrote using a set of expert-generated classification rules.

Each rule describes a specific observable behavior tied to one Big Five personality trait level {load_trait(personality_trait)}.

Your task is to predict the {load_trait(personality_trait)} level of the participant between "Low", "Moderate", "High" or "Unknwon".  Also evaluate whether the given post clearly and significantly expresses the behavior described in each rule.

Apply these principles:

Only match a rule if the behavior is strongly and clearly present.

If the behavior is weak, ambiguous, sarcastic, or forced, the rule does not apply.

Treat the rule’s behavior description, linguistic indicators, and edge cases as binding guidance.

For each rule:

Quote the specific part of the post that shows the behavior (if any)

Explain why the rule applies or doesn’t apply

Give a relevance rating (0–5) based on behavioral strength and clarity:

0 = Not at all present

1 = Extremely weak or false match

2 = Weak or ambiguous

3 = Plausible but unclear

4 = Strong match

5 = Very clear match, strongly expressed

✅ Your output must be a JSON array with one object per rule, with this exact structure:

{{
  "rule_name": "string",
  "rule_applies": true | false,
  "quoted_text": "string (quote from post or empty string)",
  "explanation": "string",
  "relevance_rating": 0–5
}}
'''

    user_content = f'''

  "post1": {post1},
  "post2": {post2},
  "rules": {rules}
  
  ✅ EXPECTED OUTPUT FORMAT
{{
  "rule_matches": [
    {{
      "rule_name": "Group Social Enthusiasm",
      "rule_applies": false,
      "quoted_text": "",
      "explanation": "The post expresses discomfort with social events, not enthusiasm. This rule does not apply.",
      "relevance_rating": 0
    }},
    {{
      "rule_name": "Preference for Solitary Activities",
      "rule_applies": true,
      "quoted_text": "I love just staying in and reading rather than going out.",
      "explanation": "This sentence expresses enjoyment of solitary activities over socializing, directly matching the rule.",
      "relevance_rating": 4
    }},
    {{
      "rule_name": "Moderate Social Engagement",
      "rule_applies": false,
      "quoted_text": "Social events just drain me.",
      "explanation": "This expresses strong negative feelings about socializing, which is excluded by the rule's edge cases.",
      "relevance_rating": 1
    }}
  ],
  "trait_classification": {{
    "trait": "{load_trait(personality_trait)}",
    "classified_level": "Low",
    "justification": "The post strongly matched the rule for Low {load_trait(personality_trait)} with a relevance rating of 5, and did not match any rules for Moderate or High."
  }}
}}'''

    return system_content, user_content
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
    system_content = f'''You are helping design structured, behavior-based classification rules for identifying the Big Five personality trait level {load_trait(personality_trait)} of a writer based on text (e.g., social media posts).

You will receive multiple psychologist-written assessments that describe behaviors correlated with a low {load_trait(personality_trait)} level.

From those assessments, extract behavioral patterns and write clear, structured classification rules.

⚠️ Another agent will later classify posts using only the rules you generate. Your output must be self-contained, specific, and consistent in structure. Rules must generalize to a variety of writing styles and contexts.

🔸 Your Task

Read and analyze the behavioral assessments.

Identify distinct, observable behavioral patterns linked to low {load_trait(personality_trait)}.

Generate a rule only if the behavior appears in at least two different assessments.

Output each rule as a JSON object in a list.

Ensure all fields are filled, follow strict structure, and avoid vague/general language.

🔹 Output Format (JSON):

[
{{
"rule_name": "Short descriptive name summarizing the behavior (e.g., 'Avoidance of Group Plans')",
"trait": "{load_trait(personality_trait)}",
"level": "Low",
"behavior_rule": "If a post contain a specific, clearly observable behavior pattern (non-redundant and unambiguous), it likely indicates low {load_trait(personality_trait)}.",
"supporting_examples": [
"Quote 1 from assessments that shows this behavior clearly",
"Quote 2 from assessments with similar behavior",
"Optional: Quote 3 (if available)"
],
"psychological_justification": "Explain how this observed behavior aligns with low {load_trait(personality_trait)} using trait theory or established psychological insight. Avoid speculation.",
"linguistic_indicators": [
"Keywords or short phrases that tend to signal this behavior (e.g., 'stayed home', 'drained by people')",
"Avoid too literal phrases—include abstract or stylistic indicators if they match the pattern"
],
"edge_cases": [
"Situations where this behavior appears but does NOT reflect low {load_trait(personality_trait)} (e.g., mentions of social fatigue after a positive event)",
"Avoid false positives from sarcasm, obligation, or non-preference-based behaviors"
],
"operational": {{
"direction": -1,
"magnitude": "integer 1–3 (strength of signal, higher=stronger)",
"confidence": "float 0.0–1.0 (specificity of indicators)",
"if_any": ["phrases that trigger the rule if ANY are found"],
"patterns": ["regex-style patterns for more flexible matches (⚠️ all backslashes must be DOUBLE escaped: use \\\\b, \\\\s, etc.)"],
"reject_if": ["phrases/contexts where the rule should not apply"],
"abstain_policy": {{
"min_abs_score": "numeric cutoff for weak evidence → abstain",
"min_evidence": "minimum number of matches"
}},
"evidence_extraction": "instructions to extract matched spans",
"tests": {{
"positives": ["2 sample sentences that SHOULD trigger the rule"],
"negatives": ["2 sample sentences that should NOT trigger the rule"]
}}
}}
}},
...
]

🔸 Constraints

Use only patterns clearly present in the provided assessments. No outside knowledge or generalizations.

A rule should be included only if it is:

Behaviorally specific

Distinct from other rules

Potentially useful for classifying unseen real-world posts

Rules should match conservatively: only if the behavioral signal is clearly evident and would likely be recognized by a human psychologist.

Limit output to 3 to 6 rules maximum.

Regex/JSON safety: In operational.patterns, all regex backslashes must be double-escaped so JSON is valid and the regex compiles (e.g., write "\\\\bquiet\\\\b", "\\\\s", not "\bquiet\b" or "\s").

🔸 Rule Part Expectations

Field	What It Should Contain
rule_name	A concise label capturing the unique behavior pattern. Avoid vague or generic titles.
trait / level	The Big Five trait (e.g., Extraversion) and the specific level this rule reflects (Low).
behavior_rule	A condition for rule application. Must start with "If a post" and must end with "then classify with " and describe a single, clear behavior (e.g., "If a post avoid initiating social contact"). Avoid compound logic or inferred emotions.
supporting_examples	2–3 direct quotes from the assessments showing this behavior. Must clearly reflect the trait level.
psychological_justification	Explain why the behavior is linked to the trait. Base it on trait theory or assessment language. Do not speculate or generalize.
linguistic_indicators	Phrases or keywords likely to appear in matching posts. Go beyond obvious words—include emotional tone, stylistic signals, or context-specific phrasing.
edge_cases	Describe cases where the rule might match but shouldn’t. Helps avoid false positives or misclassification.
operational	Machine-usable block: contains direction, magnitude, confidence, if_any, patterns, reject_if, abstain_policy, evidence_extraction, tests.

🔸 Generalization & Strengthening Guidelines

If several quotes express the same behavioral tendency, merge them into one generalized rule.

Favor stylistic and content-level signals over fixed phrases.

Avoid rules that rely solely on absence (e.g., “does not mention people”).

Ensure every rule captures a clear behavior, not just a lack of high trait behavior.

Each rule must be univariate: represent only one behavioral signal.

Avoid interpreting emotional states unless explicitly stated in the language (e.g., “I hate crowds” is okay; “seems anxious” is not).

🚫 Rule Rejection Criteria

Do not include a rule if:

It relies on vague emotional descriptions without clear behavioral grounding (e.g., “shows excitement”, “seems to enjoy company”).

It could match a large portion of neutral or positive social posts regardless of trait (e.g., birthday or family messages).

It overlaps significantly with another rule without introducing a new behavioral signal.

It merely rephrases an existing rule or expresses the same idea using different emotional terms.

It is based on inferred psychological states or attitudes not clearly observable in the text.

It could match sarcastic, humorous, or context-driven behavior without indicating a true trait signal.

It lacks identifiable linguistic cues or is too abstract for a language model to apply reliably.

Instead, prefer rules that:

Are anchored in language-level behaviors or distinct verbal styles.

Can be reasonably verified using linguistic evidence from the text alone.

Would apply only when a meaningful behavioral signal is present, and not just because the post topic is social or emotional in nature.

🔸 Final Note

Your output must be valid JSON only, with no extra explanation or headings. These rules will be parsed and applied by another AI agent without context, so clarity, behavioral grounding, and minimal ambiguity are critical. '''   
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
You are helping design structured, behavior-based classification rules for identifying the Big Five personality trait level {load_trait(personality_trait)} of a writer based on text (e.g., social media posts).

You will receive multiple psychologist-written assessments that describe behaviors correlated with a **high {load_trait(personality_trait)}** level.

From those assessments, extract behavioral patterns and write **clear, structured classification rules**.

⚠️ Another agent will later classify posts using only the rules you generate. Your output must be **self-contained**, **specific**, and **consistent in structure**. Rules must generalize to a variety of writing styles and contexts.

---

🔸 Your Task

- Read and analyze the behavioral assessments.
- Identify **distinct, observable behavioral patterns** linked to **high {load_trait(personality_trait)}**.
- Generate a rule **only if** the behavior appears in at least **two different assessments**.
- Output each rule as a **JSON object in a list**.
- Ensure **all fields are filled**, follow strict structure, and avoid vague/general language.

---

🔹 Output Format (JSON):

[
{{
"rule_name": "Short descriptive name summarizing the behavior (e.g., 'Avoidance of Group Plans')",
"trait": "{load_trait(personality_trait)}",
"level": "High",
"behavior_rule": "If a post contain a specific, clearly observable behavior pattern (non-redundant and unambiguous), it likely indicates High {load_trait(personality_trait)}.",
"supporting_examples": [
"Quote 1 from assessments that shows this behavior clearly",
"Quote 2 from assessments with similar behavior",
"Optional: Quote 3 (if available)"
],
"psychological_justification": "Explain how this observed behavior aligns with low {load_trait(personality_trait)} using trait theory or established psychological insight. Avoid speculation.",
"linguistic_indicators": [
"Keywords or short phrases that tend to signal this behavior (e.g., 'stayed home', 'drained by people')",
"Avoid too literal phrases—include abstract or stylistic indicators if they match the pattern"
],
"edge_cases": [
"Situations where this behavior appears but does NOT reflect high {load_trait(personality_trait)},
"Avoid false positives from sarcasm, obligation, or non-preference-based behaviors"
],
"operational": {{
"direction": -1,
"magnitude": "integer 1–3 (strength of signal, higher=stronger)",
"confidence": "float 0.0–1.0 (specificity of indicators)",
"if_any": ["phrases that trigger the rule if ANY are found"],
"patterns": ["regex-style patterns for more flexible matches (⚠️ all backslashes must be DOUBLE escaped: use \\\\b, \\\\s, etc.)"],
"reject_if": ["phrases/contexts where the rule should not apply"],
"abstain_policy": {{
"min_abs_score": "numeric cutoff for weak evidence → abstain",
"min_evidence": "minimum number of matches"
}},
"evidence_extraction": "instructions to extract matched spans",
"tests": {{
"positives": ["2 sample sentences that SHOULD trigger the rule"],
"negatives": ["2 sample sentences that should NOT trigger the rule"]
}}
}}
}},
...
]

---

🔸 Constraints

- Use **only** patterns clearly present in the provided assessments. No outside knowledge or generalizations.
- A rule should be included **only if** it is:
  - **Behaviorally specific**
  - **Distinct from other rules**
  - **Potentially useful for classifying unseen real-world posts**

- Rules should **match conservatively**: only if the behavioral signal is **clearly evident** and would likely be recognized by a human psychologist.
- Limit output to **3 to 6 rules** maximum.

---

🔸 Rule Part Expectations

| Field | What It Should Contain |
|-------|-------------------------|
| `rule_name` | A concise label capturing the unique behavior pattern. Avoid vague or generic titles. |
| `trait` / `level` | The Big Five trait (e.g., Extraversion) and the specific level this rule reflects (High). |
| `behavior_rule` | A condition for rule application. Must start with "If a post" and describe a single, clear behavior (e.g., "If a post avoids initiating social contact"). Avoid compound logic or inferred emotions
| `supporting_examples` | 2–3 direct quotes from the assessments showing this behavior. Must clearly reflect the trait level. |
| `psychological_justification` | Explain *why* the behavior is linked to the trait. Base it on trait theory or assessment language. Do not speculate or generalize. |
| `linguistic_indicators` | Phrases or keywords **likely to appear** in matching posts. Go beyond obvious words—include emotional tone, stylistic signals, or context-specific phrasing. |
| `edge_cases` | Describe cases where the rule might match **but shouldn’t**. Helps avoid false positives or misclassification. |
|  operational | Machine-usable block: contains direction, magnitude, confidence, if_any, patterns, reject_if, abstain_policy, evidence_extraction, tests. |

---

🔸 Generalization & Strengthening Guidelines

- If several quotes express the **same behavioral tendency**, merge them into one generalized rule.
- Favor **stylistic and content-level signals** over fixed phrases.
- Avoid rules that rely solely on **absence** (e.g., “does not express concern”).
- Ensure every rule captures a **clear behavior**, not just a lack of low trait behavior.
- Each rule must be **univariate**: represent **only one** behavioral signal.
- Avoid interpreting emotional states unless **explicitly stated** in the language (e.g., “I felt amazing at the party” is okay; “seems cheerful” is not).

---

🚫 Rule Rejection Criteria

Do **not include** a rule if:

- It relies on vague emotional descriptions without clear behavioral grounding (e.g., “shows excitement”, “seems happy with people”).
- It could match a large portion of neutral or polite social posts regardless of trait (e.g., “Happy birthday, love you!”).
- It overlaps significantly with another rule without introducing a new behavioral signal.
- It merely rephrases an existing rule or expresses the same idea using different emotional terms.
- It is based on inferred psychological states or attitudes not clearly observable in the text.
- It could match sarcastic, humorous, or context-driven behavior without indicating a true trait signal.
- It lacks identifiable linguistic cues or is too abstract for a language model to apply reliably.

Instead, prefer rules that:

- Are **anchored in language-level behaviors** or **distinct verbal styles**.
- Can be **reasonably verified** using linguistic evidence from the text alone.
- Would apply **only when a meaningful behavioral signal is present**, and **not** just because the post topic is social or emotional in nature.

---

🔸 Final Note

Your output must be valid JSON only, with no extra explanation or headings. These rules will be parsed and applied by another AI agent without context, so **clarity, behavioral grounding, and minimal ambiguity are critical**.
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

extra_prompt_shot_desc_traits = {
  "e": "Extraversion is characterized by sociability, assertiveness, talkativeness, enthusiasm, and a tendency to seek stimulation in the company of others.",
  "a": "Agreeableness is characterized by compassion, cooperation, trust in others, empathy, and a tendency to prioritize social harmony and get along well with others.",
  "c": "Conscientiousness is characterized by organization, reliability, self-discipline, attention to detail, and a strong sense of duty and goal-directed behavior.",
  "n": "Neuroticism is characterized by emotional instability, anxiety, moodiness, irritability, and a tendency to experience negative emotions more frequently and intensely.",
  "o": "Openness to Experience is characterized by curiosity, imagination, creativity, open-mindedness, and a preference for novelty, variety, and intellectual exploration."
}

# Prepare a prompt format for asking gpt to classify trait level based on two posts.
def prompt_for_shot_classification(personality_trait, post1, post2):
    system_prompt = f'''
You are an expert in personality psychology. 
Classify the level of {load_trait(personality_trait)} expressed in the following text into one of four categories: "Low", "Moderate", "High", or "Unknown".  

⚠️ Do not use the topic itself (birthday post or childhood friends gathering) to infer the level of extraversion. 
Base your decision only on the writer’s language, tone, and behavioral cues in the text.  

Return your answer only in this JSON format:

{{
  "level": "Low | Moderate | High | Unknown",
  "explanation": "Reasoning based only on linguistic or behavioral cues, not on the topic itself."
}}'''
    
    user_content = f'''Here are two social media posts written by the same person:
    Post 1: {post1}
    Post 2: {post2}''' 
    return system_prompt, user_content

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
    system_content = f'''
You are a behavioral classification agent tasked with analyzing a participant’s personality based on two social media posts and a set of expert-crafted behavioral rules.

Each rule corresponds to a specific, observable behavior that signals a particular level of the Big Five personality trait: **{load_trait(personality_trait)}**.

---

🔹 TASK

Your job is to:
1. Evaluate how well each rule applies to the posts.
2. Justify your decisions clearly with quoted evidence.
3. Rate the behavioral match strength using a 0–5 scale.
4. Assign a final trait level to the participant: **"Low"**, **"Moderate"**, **"High"**, or **"Unknown"**.

---

🔸 POST CONTEXT

You will analyze two social media posts from the same person:
- **Post 1**: A reflection about a social gathering with childhood friends.
- **Post 2**: A birthday message to the participant’s significant other.

You must base your evaluation solely on these two posts. **Do not infer personality from omissions or absence of behaviors.**

---

🔹 FOR EACH RULE:

Evaluate the content of the 2 posts against each rule’s:
- **Behavior description**
- **Linguistic indicators**
- **Edge cases**

Apply the rule **only if the behavior is clearly expressed is both of the posts** and consistent with the rule’s logic. If expression is ambiguous, weak, sarcastic, or situationally forced, the rule does **not** apply.

---

🔸 RULE MATCH FORMAT

For **each rule**, return an object with this exact structure:

```json
{{
  "rule_name": "string",
  "rule_applies": true | false,
  "quoted_text": "exact quote from post, or empty string if rule does not apply",
  "explanation": "why the rule does or does not apply, citing the rule’s logic and text content",
  "relevance_rating": integer from 0 to 5
}}
Use this relevance rating scale:

5: Behavior is clearly and strongly expressed.

4: Behavior is present but somewhat softer or less direct.

3: Behavior is implied or moderately ambiguous.

2: Behavior is weak, vague, or indirectly suggested.

1: Match is misleading, sarcastic, or contextually invalid.

0: No match whatsoever.

🔹 FINAL TRAIT CLASSIFICATION

After evaluating all rules, provide a trait-level decision using this structure:

{{
  "trait_classification": {{
    "trait": "{load_trait(personality_trait)}",
    "classified_level": "Low" | "Moderate" | "High" | "Unknown",
    "justification": "Explain how the rule matches and their relevance ratings support your classification. Mention the number of matched rules per level and whether they were strong matches (≥4)."
  }}
}}
🔸 REMINDERS

Be conservative: only apply a rule when behavior is explicitly expressed.

Edge cases in rules override any weak matches.

Use both posts as potential sources for each rule match.

Do not use outside knowledge or inference about personality traits.

Make sure all relevance ratings are meaningful — avoid overusing 0 or 5.

'''

    user_content = f'''

🧾 INPUT STRUCTURE
  
  "post1": {post1},
  "post2": {post2},
  "rules": {rules}
  
  
✅ EXPECTED OUTPUT FORMAT

  "rule_matches": [
    {{
      "rule_name": "Group Social Enthusiasm",
      "rule_applies": false,
      "quoted_text": "",
      "explanation": "The post mentions attending a gathering, but the tone is neutral and lacks enthusiasm. The rule requires clear excitement.",
      "relevance_rating": 2
    }},
    {{
      "rule_name": "Preference for Solitary Activities",
      "rule_applies": true,
      "quoted_text": "Honestly, I would’ve preferred staying home with a book.",
      "explanation": "This expresses a clear preference for solitude over socializing, matching the rule's behavioral pattern.",
      "relevance_rating": 5
    }}
  ],
  "trait_classification": {{
    "trait": "Extraversion",
    "classified_level": "Low",
    "justification": "Two Low-level rules were strongly matched (relevance ratings: 5, 4), and no rules for Moderate or High applied. The participant clearly favors solitude and shows no enthusiasm for group interaction."
  }}
}}'''
    return system_content, user_content
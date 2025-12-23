# -*- coding: utf-8 -*-
"""
Created on Sun May 18 20:43:43 2025

@author: User
"""

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

🔸 Your Task:

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
"behavior_rule": "If the person expresses / describes / shows in their post [observable behavior / linguistic pattern], then {load_trait(personality_trait)} is likely low because [brief rationale linked to the behavior itself].",
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

}}
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

Field	What It Should Contain
rule_name	A concise label capturing the unique behavior pattern. Avoid vague or generic titles.
trait / level	The Big Five trait (e.g., Extraversion) and the specific level this rule reflects (Low).
behavior_rule	A condition for rule application. Must start with "If the person expresses / describes / shows in their post" and must end with "then {load_trait(personality_trait)} is likely low because ..." and describe a single, clear behavior (e.g., "If the person shows in their post that they are staying home instead of going out, then Extraversion is likely low because this reflects limited engagement in social activities."). Avoid compound logic or inferred emotions.
supporting_examples	2–3 direct quotes from the assessments showing this behavior. Must clearly reflect the trait level.
psychological_justification	Explain why the behavior is linked to the trait. Base it on trait theory or assessment language. Do not speculate or generalize.
linguistic_indicators	Phrases or keywords likely to appear in matching posts. Go beyond obvious words—include emotional tone, stylistic signals, or context-specific phrasing.
edge_cases	Describe cases where the rule might match but shouldn’t. Helps avoid false positives or misclassification.

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

Your output must be valid JSON only, with no extra explanation or headings. These rules will be parsed and applied by another AI agent without context, so clarity, behavioral grounding, and minimal ambiguity are critical.
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

🔹 You must stick to this output format (JSON):

[
{{
"rule_name": "Short descriptive name summarizing the behavior (e.g., 'Avoidance of Group Plans')",
"trait": "{load_trait(personality_trait)}",
"level": "High",
"behavior_rule": "If the person shows / expresses / describes in their post a specific, clearly observable behavior pattern (non-redundant and unambiguous), it likely indicates High {load_trait(personality_trait)}.",
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
]
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
| behavior_rule |	A condition for rule application. Must start with "If the person expresses / describes / shows in their post" and must end with "then {load_trait(personality_trait)} is likely high because ..." and describe a single, clear behavior (e.g., "If the person shows in their post expresses enthusiasm for social interactions and group activities, then Extraversion is likely high because such expressions reflect sociability and reward-seeking."). Avoid compound logic or inferred emotions.
| `supporting_examples` | 2–3 direct quotes from the assessments showing this behavior. Must clearly reflect the trait level. |
| `psychological_justification` | Explain *why* the behavior is linked to the trait. Base it on trait theory or assessment language. Do not speculate or generalize. |
| `linguistic_indicators` | Phrases or keywords **likely to appear** in matching posts. Go beyond obvious words—include emotional tone, stylistic signals, or context-specific phrasing. |
| `edge_cases` | Describe cases where the rule might match **but shouldn’t**. Helps avoid false positives or misclassification. |

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


# Prepare a prompt format for asking gpt to classify trait level based on two posts.
def prompt_for_shot_classification(personality_trait, post1, post2):
    system_prompt = f'''
You are a behavioral classification agent tasked with analyzing a participant’s Big Five personality trait: **{load_trait(personality_trait)}** based on two social media posts.

---

🔹 TASK

Your job is to:
1. Justify your decisions clearly with quoted evidence.
2. Assign a final trait level to the participant: **"Low"**, **"Moderate"**, **"High"**, or **"Unknown"**.

---

🔸 POST CONTEXT

You will analyze two social media posts from the same person:
- **Post 1**: A reflection about a social gathering with childhood friends.
- **Post 2**: A birthday message to the participant’s significant other.

You must base your evaluation solely on these two posts. **Do not infer personality from omissions or absence of behaviors.**

---

Return your answer only in this JSON format:

{{
  "level": "Low | Moderate | High | Unknown",
  "explanation": "Reasoning based only on linguistic or behavioral cues, not on the topic itself."
}}'''
    
    user_content = f'''🧾 INPUT STRUCTURE
      
      post1: {post1},
      post2: {post2}
'''
    return system_prompt, user_content


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

If expression is ambiguous, weak, sarcastic, or situationally forced, the rule does **not** apply.

---

🔸 RULE MATCH FORMAT


For each rule match, you must quote the texts that made you match. The quotes must be contained in square brackets to form a valid python list. 

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
    "justification": "Explain why and how the rules match and their relevance ratings support your classification."
  }}
}}
🔸 REMINDERS

Be conservative: only apply a rule when behavior is explicitly expressed.

Edge cases in rules override any weak matches.

Use both posts as potential sources for each rule match.

Do not use outside knowledge or inference about personality traits.

Make sure all relevance ratings are meaningful — avoid overusing 0 or 5.

rules: {rules}

'''

    user_content = f'''

🧾 INPUT STRUCTURE
  
  post1: {post1},
  post2: {post2}
  
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
      "quoted_text": ["Honestly, I would’ve preferred staying home with a book.",  "I would prefer hanging out alone"],
      "explanation": "This expresses a clear preference for solitude over socializing, matching the rule's behavioral pattern.",
      "relevance_rating": 5
    }}
  ],
  "trait_classification": {{
    "trait": "Extraversion",
    "classified_level": "Low",
    "justification": "string"
  }}
}}'''
    return system_content, user_content
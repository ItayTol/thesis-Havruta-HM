# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 23:33:55 2025

@author: User
"""

# We'll create a new self-contained hyperparam tuning runner that:
# - defines build_prompt_with_variant(personality_trait, post1, post2, rules, variant)
# - uses your existing OpenAI wrapper: havruta_HM_thesis_gpt_response_3_classes.standard_gpt_response
# - runs a tiny grid search over variants on a provided validation dataframe (schema: ['post1','post2','rules','true_label'])
# - is careful to only require your existing modules and to avoid the buggy duplicate in classification_prep.
# We'll write it to /mnt/data/havruta_HM_hyperparams_grid.py

# Hyperparameter prompt-variant grid search (prompt-only) for 3-class classification.
# Assumes availability of:
#  - havruta_HM_thesis_gpt_response_3_classes.standard_gpt_response (OpenAI API wrapper)
#  - A validation DataFrame with columns: ['post1', 'post2', 'rules', 'true_label']

import json
import hashlib
from typing import Dict, Tuple, List
import pandas as pd
from string import Template

from havruta_HM_thesis_gpt_response_3_classes import standard_gpt_response
from havruta_HM_thesis_prompts_3_classes import load_trait

# -----------------------------
# Prompt Builder
# -----------------------------

# 7-9-25 suggestion ------------------------

SYSTEM_TMPL = Template(r"""
You are a behavioral classification agent. You analyze two short social-media posts by the same person using expert-crafted behavioral rules for the target Big Five trait **$TRAIT_NAME**.

Your goals:
1) Evaluate how well each rule applies to the posts;
2) Justify with **short** quoted evidence taken from te participant's posts(=< 20 words each, in [brackets]);
3) Rate behavioral match strength on a 0–5 scale (see anchors);
4) Produce a final trait level: "Low", "Moderate", "High", or "Unknown".

INPUTS
- Post 1: a reflection about a social gathering with childhood friends.
- Post 2: a birthday message to the significant other.
- Rules: structured items with behavior description, linguistic indicators, edge cases, magnitude (1–3), confidence (0–1), and target level.

GENERAL CONDUCT
- Use only the two posts. Do not infer from omissions.
- Be conservative: if evidence is ambiguous, choose the lower relevance rating.
- Edge cases in a rule override weak matches.
- Keep quotes minimal and exact; prefer literal phrases from the participant's posts.

RATING ANCHORS (interpret consistently)
- 5: Behavior is explicit, central, and clearly fits the rule.
- 4: Behavior is present and fairly clear, but softer/less central.
- 3: Behavior is implied or moderately ambiguous.
- 2: Behavior is weak/vague/indirect.
- 1: Behavior is misleading, sarcastic, or contextually invalid.
- 0: No match.

RULE EVALUATION FORMAT (per rule)
Return for each rule:
{
  "rule_name": "...",
  "rule_applies": true|false,
  "quoted_text": ["[...]", "..."] or "",
  "explanation": "...",
  "relevance_rating": 0..5,
  "rule_score": <numeric>,         // In the JSON output, "rule_score" must always be a numeric literal (already calculated), never a formula or expression.
  "target_level": "Low"|"High"       // the rule’s own level
}

SCORING & DECISION POLICY (HYPERPARAMS)
You are running prompt variant: $VARIANT_ID.

Use only rules with rule_confidence => $CONFIDENCE_CUTOFF.

EVIDENCE POLICY:
A rule “applies” only if relevance_rating => $RELEVANCE_CUTOFF and the behavior $EVIDENCE_REQUIREMENT.

RULE SCORE:
For each applied rule, compute
  rule_score = $MAGNITUDE_MODE(magnitude) × rule_confidence × $STRONG_MATCH_BONUS(relevance_rating).
If $CAP_TOP_RULES_PER_LEVEL is an integer, keep at most that many highest rule_score contributors **per level**; drop the rest from aggregation.

AGGREGATION:
Sum rule_score by the rule’s target level → totals.low, totals.high.

UNKNOWN POLICY:
If $UNKNOWN_CONDITION then set classified_level = "Unknown".

MODERATE BACKSTOP:
If neither totals.low nor totals.high exceeds the other by => $MODERATE_BACKSTOP_DELTA × max(totals.low, totals.high)
AND at least one non-trivial rule applies (any relevance_rating => 3),
then set classified_level = "Moderate".

FINAL DECISION (if not Unknown or Backstop):
- If $AGGREGATION_METHOD == "majority": choose the level with the most applied rules (ties → higher totals.*).
- If $AGGREGATION_METHOD == "weighted_sum": choose the level with the largest totals.*.

OUTPUT (JSON only)
- "variant_id": "$VARIANT_ID"
- "hyperparams": {
    "relevance_cutoff": $RELEVANCE_CUTOFF,
    "confidence_cutoff": $CONFIDENCE_CUTOFF,
    "evidence_requirement": "$EVIDENCE_REQUIREMENT",
    "aggregation_method": "$AGGREGATION_METHOD",
    "unknown_condition": "$UNKNOWN_CONDITION",
    "magnitude_mode": "$MAGNITUDE_MODE",
    "strong_match_bonus": "$STRONG_MATCH_BONUS",
    "cap_top_rules_per_level": $CAP_TOP_RULES_PER_LEVEL,
    "moderate_backstop_delta": $MODERATE_BACKSTOP_DELTA
  }
- "rule_matches": [ ... as above ... ]
- "applied_rules_count": <int>,
- "strong_applied_rules_count": <int with relevance_rating >= 4>,
- "totals": {"low": <num>, "high": <num>},
- "unknown_triggered": true|false,
- "trait_classification": {
    "trait": "$TRAIT_NAME",
    "classified_level": "Low"|"Moderate"|"High"|"Unknown",
    "justification": "Cite key rules, relevance ratings, quotes from posts to prove the classified personality level, and totals.* that drove the decision."
}
""")

USER_TMPL = Template(r"""
🧾 INPUT STRUCTURE
  "post1": $POST1,
  "post2": $POST2,
  "rules": $RULES

✅ EXPECTED OUTPUT FORMAT
  "rule_matches": [...],
  "applied_rules_count": <int>,
  "strong_applied_rules_count": <int>,
  "totals": {"low": <num>, "high": <num>},
  "unknown_triggered": true|false,
  "trait_classification": {
    "trait": "$TRAIT_NAME",
    "classified_level": "Low" | "Moderate" | "High" | "Unknown",
    "justification": "..."
  }
""")

VARIANTS = [

  # Balanced default: one-post allowed (no contradiction), linear magnitude
  dict(
    variant_id="vA",
    RELEVANCE_CUTOFF=3,
    CONFIDENCE_CUTOFF=0.80,
    EVIDENCE_REQUIREMENT="appears in at least one post with relevance_rating >= 4 and is not contradicted in the other post (rating <= 1 on opposing behavior)",
    AGGREGATION_METHOD="weighted_sum",
    UNKNOWN_CONDITION="max(totals.low, totals.high) < 0.8",
    MAGNITUDE_MODE="m",
    STRONG_MATCH_BONUS="1.2 if r>=4 else 1.0",
    CAP_TOP_RULES_PER_LEVEL=3,
    MODERATE_BACKSTOP_DELTA=0.12
  ),

  # Majority voting version of vA (more robust when a few medium rules fire)
  dict(
    variant_id="vB",
    RELEVANCE_CUTOFF=3,
    CONFIDENCE_CUTOFF=0.80,
    EVIDENCE_REQUIREMENT="appears in at least one post with relevance_rating >= 4 and is not contradicted",
    AGGREGATION_METHOD="majority",
    UNKNOWN_CONDITION="max(totals.low, totals.high) < 0.9",
    MAGNITUDE_MODE="m",
    STRONG_MATCH_BONUS="1.2 if r>=4 else 1.0",
    CAP_TOP_RULES_PER_LEVEL=3,
    MODERATE_BACKSTOP_DELTA=0.12
  ),

  # Slightly stricter relevance; still allows one strong post; linear magnitude
  dict(
    variant_id="vC",
    RELEVANCE_CUTOFF=4,
    CONFIDENCE_CUTOFF=0.80,
    EVIDENCE_REQUIREMENT="appears in both posts OR appears in at least one post with relevance_rating >= 5 without contradiction",
    AGGREGATION_METHOD="weighted_sum",
    UNKNOWN_CONDITION="max(totals.low, totals.high) < 1.0",
    MAGNITUDE_MODE="m",
    STRONG_MATCH_BONUS="1.3 if r>=5 else 1.0",
    CAP_TOP_RULES_PER_LEVEL=3,
    MODERATE_BACKSTOP_DELTA=0.12
  ),

  # Magnitude-squared stress test, but tamed bonus and softer confidence
  dict(
    variant_id="vD",
    RELEVANCE_CUTOFF=3,
    CONFIDENCE_CUTOFF=0.75,
    EVIDENCE_REQUIREMENT="appears in at least one post with relevance_rating >= 4 and is not contradicted",
    AGGREGATION_METHOD="weighted_sum",
    UNKNOWN_CONDITION="max(totals.low, totals.high) < 1.0",
    MAGNITUDE_MODE="m^2",
    STRONG_MATCH_BONUS="1.3 if r>=4 else 1.0",
    CAP_TOP_RULES_PER_LEVEL=2,
    MODERATE_BACKSTOP_DELTA=0.10
  ),

  # Conservative evidence (both posts), but lower cutoff so rules actually fire
  dict(
    variant_id="vE",
    RELEVANCE_CUTOFF=3,
    CONFIDENCE_CUTOFF=0.80,
    EVIDENCE_REQUIREMENT="appears in both posts",
    AGGREGATION_METHOD="weighted_sum",
    UNKNOWN_CONDITION="(applied_rules_count == 0) or (max(totals.low, totals.high) < 0.8)",
    MAGNITUDE_MODE="m",
    STRONG_MATCH_BONUS="1.0",
    CAP_TOP_RULES_PER_LEVEL=4,
    MODERATE_BACKSTOP_DELTA=0.15
  ),

  # Tie-friendly, majority vote + moderate backstop; good when Low/High clash
  dict(
    variant_id="vF",
    RELEVANCE_CUTOFF=3,
    CONFIDENCE_CUTOFF=0.80,
    EVIDENCE_REQUIREMENT="appears in at least one post with relevance_rating >= 4 and is not contradicted",
    AGGREGATION_METHOD="majority",
    UNKNOWN_CONDITION="max(totals.low, totals.high) < 0.8",
    MAGNITUDE_MODE="m",
    STRONG_MATCH_BONUS="1.0",
    CAP_TOP_RULES_PER_LEVEL=3,
    MODERATE_BACKSTOP_DELTA=0.18
  ),
]



def build_prompt_with_variant_suggestion(trait_name:str, post1:str, post2:str, rules:List[dict], variant:Dict) -> Tuple[str,str]:
    """Fill the master prompt template with the chosen variant knobs and the concrete inputs."""
    system_content = SYSTEM_TMPL.substitute(
        TRAIT_NAME=trait_name,
        VARIANT_ID=variant["variant_id"],
        CONFIDENCE_CUTOFF=variant["CONFIDENCE_CUTOFF"],
        RELEVANCE_CUTOFF=variant["RELEVANCE_CUTOFF"],
        EVIDENCE_REQUIREMENT=variant["EVIDENCE_REQUIREMENT"],
        AGGREGATION_METHOD=variant["AGGREGATION_METHOD"],
        UNKNOWN_CONDITION=variant["UNKNOWN_CONDITION"],
        MAGNITUDE_MODE=variant["MAGNITUDE_MODE"],
        STRONG_MATCH_BONUS=variant["STRONG_MATCH_BONUS"],
        CAP_TOP_RULES_PER_LEVEL=variant["CAP_TOP_RULES_PER_LEVEL"],
        MODERATE_BACKSTOP_DELTA=variant["MODERATE_BACKSTOP_DELTA"],
    )
    user_content = USER_TMPL.substitute(
        POST1=json.dumps(post1, ensure_ascii=False),
        POST2=json.dumps(post2, ensure_ascii=False),
        RULES=json.dumps(rules, ensure_ascii=False),
        TRAIT_NAME=trait_name,
    )
    return system_content, user_content



# ------------------------------------------


BASE_SYSTEM_PROMPT = Template(r"""\
You are a behavioral classification agent tasked with analyzing a participant’s personality based on two social media posts and a set of expert-crafted behavioral rules.

Each rule corresponds to a specific, observable behavior that signals a particular level of the Big Five personality trait: **$trait_name**.

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
    "trait": "$trait_name",
    "classified_level": "Low" | "Moderate" | "High" | "Unknown",
    "justification": "Explain how the rule matches and their relevance ratings support your classification. Mention the number of matched rules per level and whether they were strong matches (=>4)."
  }}
}}
🔸 REMINDERS

Be conservative: only apply a rule when behavior is explicitly expressed.
Edge cases in rules override any weak matches.
Use both posts as potential sources for each rule match.
Do not use outside knowledge or inference about personality traits.
Make sure all relevance ratings are meaningful — avoid overusing 0 or 5.
Quoted texts must be contained in square brackets.

You are an assistant that outputs **only valid JSON**.
Do not include explanations, comments, or markdown formatting.
Do not include trailing commas.
All strings must use double quotes.
Escape backslashes and quotes properly.
Ensure the output is strictly valid JSON that can be parsed by Python's json.loads().

---
🔹 VARIANT + LOGGING (do not skip)
You are running prompt variant: $VARIANT_ID.

Only use rules with rule_confidence => $CONFIDENCE_CUTOFF.
A rule “applies” only if relevance_rating => $RELEVANCE_CUTOFF and the behavior $EVIDENCE_REQUIREMENT.

For each applied rule, compute:
  rule_score = $MAGNITUDE_MODE(magnitude) × rule_confidence × $STRONG_MATCH_BONUS(relevance_rating)
In the JSON output, "rule_score" must always be a numeric literal (already calculated), never a formula or expression.

Aggregation:
  Sum rule_score into totals by the rule’s target level → totals.low, totals.high.

Unknown policy:
  If $UNKNOWN_CONDITION, set classified_level = "Unknown".
  Otherwise:
    - If $AGGREGATION_METHOD == "majority": choose the level with the most applied rules.
    - If $AGGREGATION_METHOD == "weighted_sum": choose the level with the largest totals.*.

In the JSON output you MUST include (in addition to your current fields):
- "variant_id": "$VARIANT_ID"
- "hyperparams": {{
    "relevance_cutoff": $RELEVANCE_CUTOFF,
    "confidence_cutoff": $CONFIDENCE_CUTOFF,
    "evidence_requirement": "$EVIDENCE_REQUIREMENT",
    "aggregation_method": "$AGGREGATION_METHOD",
    "unknown_condition": "$UNKNOWN_CONDITION",
    "magnitude_mode": "$MAGNITUDE_MODE",
    "strong_match_bonus": "$STRONG_MATCH_BONUS"
  }}
- For every rule item: add "rule_score": <numeric>, and "target_level": "Low"|"Moderate"|"High".
- Also add:
  "applied_rules_count": <int>,
  "strong_applied_rules_count": <int rules with relevance_rating >= 4>,
  "totals": {{"low": <num>, "moderate": <num>, "high": <num>}},
  "unknown_triggered": true|false.

Return strict JSON only. Do not include code blocks, comments, math expressions, or natural-language conditions inside fields.
All numeric fields (e.g., rule_score, magnitude, confidence, totals) must be numbers, not strings or expressions.""")

BASE_USER_TEMPLATE = Template(r"""\
🧾 INPUT STRUCTURE
  "post1": $post1,
  "post2": $post2,
  "rules": $rules

✅ EXPECTED OUTPUT FORMAT
  "rule_matches": [...],
  "trait_classification": {{
    "trait": "$trait_name",
    "classified_level": "Low" | "Moderate" | "High" | "Unknown",
    "justification": "..."
  }}
""")

def build_prompt_with_variant(trait_name:str, post1:str, post2:str, rules:List[dict], variant:Dict) -> Tuple[str,str]:
    """Fill the master prompt template with the chosen variant knobs and the concrete inputs."""
    system_prompt = BASE_SYSTEM_PROMPT.substitute(
        trait_name=load_trait(trait_name),
        VARIANT_ID=variant["variant_id"],
        RELEVANCE_CUTOFF=variant["RELEVANCE_CUTOFF"],
        CONFIDENCE_CUTOFF=variant["CONFIDENCE_CUTOFF"],
        EVIDENCE_REQUIREMENT=variant["EVIDENCE_REQUIREMENT"],
        AGGREGATION_METHOD=variant["AGGREGATION_METHOD"],
        UNKNOWN_CONDITION=variant["UNKNOWN_CONDITION"],
        MAGNITUDE_MODE=variant["MAGNITUDE_MODE"],
        STRONG_MATCH_BONUS=variant["STRONG_MATCH_BONUS"],
    )
    user_prompt = BASE_USER_TEMPLATE.substitute(
        post1=json.dumps(post1, ensure_ascii=False),
        post2=json.dumps(post2, ensure_ascii=False),
        rules=json.dumps(rules, ensure_ascii=False),
        trait_name=load_trait(trait_name)
    )
    return system_prompt, user_prompt

# -----------------------------
# Variants
# -----------------------------

VARIANTS_former = [
    dict(variant_id="v01",
          RELEVANCE_CUTOFF=2, CONFIDENCE_CUTOFF=0.80,
          EVIDENCE_REQUIREMENT="appears in both posts",
          AGGREGATION_METHOD="weighted_sum",
          UNKNOWN_CONDITION="no rule with relevance_rating => 4",
          MAGNITUDE_MODE="m", STRONG_MATCH_BONUS="1"),
    dict(variant_id="v02",
          RELEVANCE_CUTOFF=2, CONFIDENCE_CUTOFF=0.90,
          EVIDENCE_REQUIREMENT="appears in both posts",
          AGGREGATION_METHOD="majority",
          UNKNOWN_CONDITION="fewer than 2 applied rules",
          MAGNITUDE_MODE="m^2", STRONG_MATCH_BONUS="1.5 if r=>4 else 1"),
    dict(variant_id="v03",
          RELEVANCE_CUTOFF=3, CONFIDENCE_CUTOFF=0.80,
          EVIDENCE_REQUIREMENT="appears in at least one post with relevance_rating => 4",
          AGGREGATION_METHOD="weighted_sum",
          UNKNOWN_CONDITION="no rule with relevance_rating => 4",
          MAGNITUDE_MODE="m", STRONG_MATCH_BONUS="1"),
    dict(variant_id="v04",
         RELEVANCE_CUTOFF=3, CONFIDENCE_CUTOFF=0.80,
         EVIDENCE_REQUIREMENT="appears in at least one post with relevance_rating => 4",
         AGGREGATION_METHOD="weighted_sum",
         UNKNOWN_CONDITION="fewer than 2 applied rules",
         MAGNITUDE_MODE="m^2", STRONG_MATCH_BONUS="1.5 if r=>4 else 1"),
]

# -----------------------------
# LLM call with caching
# -----------------------------

_CACHE = {}

def _hash_key(system_prompt, user_prompt):
    return hashlib.sha256((system_prompt + "\n---\n" + user_prompt).encode("utf-8")).hexdigest()

def llm_call(system_prompt:str, user_prompt:str) -> dict:
    key = _hash_key(system_prompt, user_prompt)
    if key in _CACHE:
        return _CACHE[key]
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
        for rm in parsed.get("rule_matches", []):
            rs = rm.get("rule_score")
            if isinstance(rs, str):
                try:
                    rm["rule_score"] = eval(rs, {"__builtins__": None}, {})
                except Exception:
                    rm["rule_score"] = None

    except Exception as e:
        # Surface the bad output in the exception to debug prompt errors quickly
        raise ValueError(f"Model did not return valid JSON. \n{text}") from e
    _CACHE[key] = parsed
    return parsed

# -----------------------------
# Evaluation helpers
# -----------------------------

def evaluate_variant_on_df(variant:Dict, trait_name:str, val_df:pd.DataFrame, rules:str) -> pd.DataFrame:
    """Run a single variant on a validation dataframe and return per-row predictions + success flag."""
    preds, trues, oks, errors, outs = [], [], [], [], []
    for i, row in val_df.iterrows():
        try:
            sys_p, usr_p = build_prompt_with_variant_suggestion(trait_name, row["post1"], row["post2"], rules, variant)
            true = row[f"{trait_name}"]
            out = llm_call(sys_p, usr_p)
            intented_out = json.dumps(out, indent=2)
            pred = out["trait_classification"]["classified_level"]
            ok = (pred == true)
            preds.append(pred); trues.append(true); oks.append(int(ok)); errors.append(""); outs.append(intented_out)
        except Exception:
            preds.append(None); trues.append(true); oks.append(0); errors.append('err'); outs.append(None)

    return pd.DataFrame({
        "variant_id": variant["variant_id"],
        "p":val_df["p"],
        "post1": val_df["post1"],
        "post2": val_df["post2"],
        "pred": preds,
        "true": trues,
        "ok": oks,
        "error": errors,
        "out": outs
    })


# THIS FUNCTION IS IN THE MAIN

# def grid_search_variants(trait_name:str, val_df:pd.DataFrame, rules:str, variants:List[Dict]=None) -> pd.DataFrame:
#     if variants is None:
#         variants = VARIANTS
#     frames = []
#     for v in variants:
#         dfv = evaluate_variant_on_df(v, trait_name, val_df, rules)
#         dfv["trait_name"] = trait_name
#         frames.append(dfv)
#     big = pd.concat(frames, ignore_index=True)
#     summary = big.groupby("variant_id", as_index=False)["ok"].mean().rename(columns={"ok":"accuracy"})
#     # Attach error rate for visibility
#     summary2 = big.groupby("variant_id", as_index=False)["ok"].sum().rename(columns={"ok":"num_of_accurate"})
#     err = big.assign(err_flag=big["error"].astype(str).str.len()>0).groupby("variant_id", as_index=False)["err_flag"].sum().rename(columns={"err_flag":"error_rate"})
#     return summary.merge(err, on="variant_id").sort_values(["accuracy","error_rate"], ascending=[False, True]).merge(summary2, on="variant_id").sort_values(["accuracy","error_rate"], ascending=[False, True])
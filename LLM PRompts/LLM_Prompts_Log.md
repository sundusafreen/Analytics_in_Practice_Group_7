# LLM Prompts Log

**Project:** E-Commerce Customer Churn Prediction  
**Team Members (LLM Role):** Lavanya Batra, Konstantinos Fragkos  
**Tool Used:** [e.g. Claude by Anthropic / ChatGPT — update as needed]

> All AI outputs were manually reviewed and validated by team members before use.  
> AI was used as an assistive tool; no output was accepted without human oversight.  
> Acknowledged per [TCD GenAI referencing guidelines](https://libguides.tcd.ie/genai/acknowledging-referencing).

---

## Prompt 1 — Business Explanation of Churn Drivers

**Purpose:** Translate technical model findings into plain language for a non-technical audience.

**Prompt:**
> "Explain churn drivers for a non-technical CEO (no jargon)"

**Context provided to model:**
> [Add the feature importance data or summary stats you passed in — e.g. the top 5 coefficients from the logistic regression]

**Output summary:**
> [Paste or summarise the AI's response here]

**Validation & critique:**
- Outputs were manually verified for accuracy against model results
- Avoided over-reliance — human team members reviewed all explanations before including in slides
- [Note any inaccuracies found or corrections made]

**Used in:** Slide 7 (Business Recommendations) / Slide 1 (Problem Statement)

---

## Prompt 2 — Python Preprocessing Pipeline

**Purpose:** Generate a reproducible preprocessing pipeline for the churn dataset.

**Prompt:**
> "Write a complete Python preprocessing pipeline for a churn dataset..."

**Context provided to model:**
> [Add dataset description or column names you included in the prompt]

**Output summary:**
> [Paste or summarise the AI's response here]

**Validation & critique:**
- Identified a data leakage issue in the scaling step — StandardScaler was being fit on the full dataset before the train/test split; corrected to fit on training data only
- Code was reviewed and tested by the Modelers/Analysts team before use
- [Note any other issues found or corrections made]

**Used in:** `notebooks/ML1_EDA_LogisticRegression_v2.py` / `notebooks/ML2_RandomForest_XGBoost.py`

---

## Prompt 3 — Actionable Business Recommendations from Feature Importance

**Purpose:** Convert model feature importance scores into concrete, business-facing action steps.

**Prompt:**
> "Summarise feature importance into actionable business steps"

**Context provided to model:**
> [Add the feature importance table or chart you passed in — e.g. SHAP values or logistic regression coefficients]

**Output summary:**
> [Paste or summarise the AI's response here]

**Validation & critique:**
- Outputs were manually verified for accuracy
- Ensured recommendations were grounded in actual model findings, not generic advice
- [Note any inaccuracies found or corrections made]

**Used in:** Slide 7 (Business Recommendations)

---

## Overall Reflection

| | |
|---|---|
| **What worked well** | AI significantly reduced development time and improved clarity of insights for non-technical slides |
| **What required caution** | Data leakage issue identified in scaling step; required human correction before use |
| **Key lesson** | AI is a powerful assistant but human oversight is essential — especially for technical decisions like preprocessing order and citation accuracy |
| **Over-reliance risk** | Mitigated by always validating outputs against actual data and model results before use |

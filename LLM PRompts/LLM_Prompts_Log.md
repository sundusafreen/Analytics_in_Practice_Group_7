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
country_pakistan:          0.694
preferred_category_home:   0.665
country_india:             0.642
country_usa:               0.625
preferred_category_electronics: 0.572
quantity:                 -0.205
unit_price:               -0.136

**Output summary:**
> Our analysis of 2,000 customers shows that where a customer is based and what they shop for are the two biggest signals that they might stop buying from us.
Customers based in India and Pakistan are leaving at higher rates than anywhere else — roughly 29% and 28% respectively, compared to our overall average of 25%. This suggests there may be unmet expectations around delivery, pricing, or product availability in those markets that we are not currently addressing.
Customers who mainly buy Home and Electronics products are also more likely to leave than those who shop in other categories. These tend to be higher-consideration purchases — customers may be comparing more carefully and finding better deals elsewhere.
On the positive side, customers who buy more items and spend more per order are actually our most loyal. This tells us that high-value customers are worth protecting — a small retention investment in this group could have an outsized return.
Bottom line: Focus retention efforts on customers in India and Pakistan, and those browsing Home and Electronics. Reward your high-spending customers before they have a reason to leave.

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
> Dataset: E-Commerce Customer Insights and Churn
Columns: order_id, customer_id, age, product_id, country, 
signup_date, last_purchase_date, cancellations_count, 
subscription_status, order_date, unit_price, quantity, 
purchase_frequency, preferred_category, product_name, 
category, gender
Target variable: churn (derived from subscription_status == 'cancelled')
2,000 rows, transaction-level data to be aggregated per customer

**Output summary:**
> The AI generated a preprocessing pipeline covering date parsing, customer-level aggregation, feature engineering (tenure days, days since last purchase), one-hot encoding of categorical variables, SMOTE for class imbalance, and StandardScaler for normalisation. The pipeline structure was largely correct but contained a data leakage error — the scaler was fitted on the full dataset before the train/test split rather than on the training fold only. This was identified and corrected by the modelling team.

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

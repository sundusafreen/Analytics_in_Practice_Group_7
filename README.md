# E-Commerce Customer Churn Prediction

**Module:** Business Analytics — Group Project  
**Institution:** Trinity College Dublin  
**Deadline:** 16 April 2025  
**Dataset:** [E-Commerce Customer Insights and Churn Dataset](https://www.kaggle.com/datasets/nabihazahid/e-commerce-customer-insights-and-churn-dataset) (Kaggle, nabihazahid)

---

## Business Problem

1 in 4 customers on this e-commerce platform stops purchasing. This project builds an interpretable machine learning pipeline to identify which customers are most at risk of churning, quantify the revenue impact, and recommend targeted retention interventions with measurable ROI.

**Primary metric:** ROC-AUC on held-out test set  
**Secondary metric:** Top three churn-driving features identified and explained  
**Business impact metric:** Estimated revenue saved by targeting the top 20% highest-risk customers

---

## Repository Structure

```
├── data/
│   └── E_Commerce_Customer_Insights_and_Churn_Dataset.csv   # Dataset subset
├── notebooks/
│   ├── ML1_EDA_LogisticRegression_v2.py                     # EDA + Logistic Regression baseline
│   ├── ML2_churn_explainable_pipeline.ipynb                 # Random Forest + XGBoost models
├── charts/
│   ├── chart1_churn_distribution.png
│   ├── chart2_churn_by_country.png
│   ├── chart3_churn_by_category.png
│   ├── chart4_feature_distributions.png
│   ├── chart5_correlation_heatmap.png
│   ├── chart6_confusion_matrix_lr.png
│   ├── chart7_roc_curve_lr.png
│   └── chart8_feature_importance_lr.png
├── presentation/
│   └── Group_Churn_Presentation.pptx                        # Final 10-slide deck
├── LLM PRompts/
│   └── LLM_Prompts_Log.md                                   # All prompts, inputs, and outputs
└── README.md
```

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

> **Note:** SHAP 0.51.0 is required for XGBoost 3.x compatibility.  
> Run `pip install shap==0.51.0` if you encounter version errors.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/<(https://github.com/sundusafreen/Analytics_in_Practice_Group_7)>
   cd <repo-name>
   ```

2. Place the dataset in the `data/` folder (or update the file path in the notebook).

3. Run the EDA and Logistic Regression baseline:
   ```bash
   python notebooks/ML1_EDA_LogisticRegression_v2.py
   ```

4. Run the Random Forest and XGBoost models:
   ```bash
   python notebooks/ML2_churn_explainable_pipeline.py
   ```

All charts will be saved automatically to the `charts/` folder.

---

## Key Findings

- **Overall churn rate:** 24.6% (493 out of 2,000 customers)
- **Strongest churn drivers:** Country (India 29.3%, Pakistan 28.0%) and preferred product category (Home 28.6%, Electronics 25.4%)
- **Loyal customer signal:** Customers with higher purchase quantity and higher unit price are less likely to churn
- **Model performance:** All three models (Logistic Regression, Random Forest, XGBoost) converge to a ROC-AUC of ~0.53, consistent with near-zero feature-churn correlations in this synthetic dataset
- **Best model:** XGBoost (ROC-AUC 0.5382); recommended for per-customer SHAP waterfall explanations

---

## Models Trained

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.38 | 0.25 | 0.78 | 0.38 | 0.536 |
| Random Forest | 0.7050 | 0.3208 | 0.1717 | 0.2237 | 0.5303 |
| XGBoost | 0.6950 | 0.3380 | 0.2424 | 0.2824 | 0.5382 |

> The Logistic Regression baseline was configured with `class_weight={0:1, 1:3}` to prioritise recall — catching true churners matters more than minimising false alarms in a retention use case.

---

## Team Roles

| Name | Role |
|---|---|
| Duygu Barutcu | Data Engineer |
| Lin Htet Aung | Data Engineer |
| Weiyi Yan | Modeler / Analyst |
| Cesc Gomez | Modeler / Analyst |
| Lavanya Batra | LLM / Prompt Specialist |
| Konstantinos Fragkos | LLM / Prompt Specialist |
| Bin Yu | Visualization Expert |
| Ummea Salma Hossain | Visualization Expert |
| Sundus Afreen | Project Manager / Storyteller |
| Naija Budhiraja | Project Manager / Storyteller |

---

## LLM Use

All prompts, model inputs/outputs, and critical reflections are logged in `notebooks/LLM_Prompts_Log.md`.

Three core prompt categories were used:
1. **Prompt 1** — [placeholder: describe what this prompt was used for]
2. **Prompt 2** — Feature importance interpretation using logistic regression coefficients
3. **Prompt 3** — [placeholder: describe what this prompt was used for]

> **GenAI Acknowledgement:** Generative AI tools (including Claude by Anthropic) were used in this project for code generation, insight synthesis, and document drafting. All AI-generated content was reviewed and edited by team members. AI-generated citations were manually verified. Use is acknowledged at slide level in the presentation, in accordance with [TCD GenAI referencing guidelines](https://libguides.tcd.ie/genai/acknowledging-referencing).

---

## References

> All references were manually verified by team members. AI-generated citations were checked for accuracy before inclusion, per TCD guidelines.

- [1] Reichheld, F. F., & Schefter, P. (2000). E-loyalty: Your secret weapon on the web. *Harvard Business Review*, 78(4), 105–113.
- [2] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, San Francisco, CA, pp. 785–794.
- [3] Alshamsi, A. et al. (2024). E-commerce churn prediction for analyzing customer behaviour based on machine learning. In *Lecture Notes in Business Information Processing*. Springer.
- [4] Hu, W. et al. (2019). A hybrid prediction model for e-commerce customer churn based on logistic regression and extreme gradient boosting algorithm. *ISI Journal*, 24(5).
- [5] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4766–4777.
- [6] Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.
- Zahid, N. (n.d.). *E-Commerce Customer Insights and Churn Dataset*. Kaggle. https://www.kaggle.com/datasets/nabihazahid/e-commerce-customer-insights-and-churn-dataset

---

## Citation & Academic Integrity

This project follows TCD referencing guidelines: https://libguides.tcd.ie/c.php?g=667784&p=4736271  
GenAI referencing guidelines: https://libguides.tcd.ie/genai/acknowledging-referencing

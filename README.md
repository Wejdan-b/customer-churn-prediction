# Customer Churn Prediction & Data Mining

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wejdan-b/customer-churn-prediction/blob/main/customer_churn_prediction.ipynb)

An end-to-end data science project that analyzes **7,043 telecom customers** to uncover the patterns behind customer churn and builds a model that flags customers who are likely to leave, turning the results into practical retention recommendations.

## The Problem

Customer churn means a customer stops doing business with a company. Keeping an existing customer is usually cheaper than acquiring a new one, so knowing **who is likely to leave, before they leave**, lets the company act early.

This project answers two questions:

1. **Data Mining:** What patterns are associated with churn across groups of customers?
2. **Machine Learning:** Can we predict whether a specific customer will churn?

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d): 7,043 customers and 21 columns covering demographics, subscribed services, account information, and the target column `Churn` (Yes/No).

## Tools

Python · Pandas · Matplotlib · Seaborn · Scikit-learn · Google Colab

## Workflow

### 1. Data Cleaning
- Found that `TotalCharges` was stored as text (`object`) instead of a number.
- Investigated and found **11 customers** whose value was a blank space; all had `tenure = 0` (new customers who had not been billed yet).
- Filled them with `0.0` (their true value) rather than dropping real customers, then converted the column to `float`.
- Checked for missing values and duplicate rows (none found).

### 2. Exploratory Data Analysis
Followed an **observation → hypothesis → verification** approach to find churn patterns.

![Churn Rate by Contract Type](<img width="576" height="455" alt="churn-contract" src="https://github.com/user-attachments/assets/c2412582-c12c-4162-89da-1fc961d472d7" />
)

### 3. Data Preparation
- Dropped `customerID` (no behavioral information).
- Encoded the target (`Yes` = 1, `No` = 0) and applied **One-Hot Encoding** to categorical features.
- Split the data **80/20** with `stratify` to keep the churn ratio equal in both sets.
- Scaled features with `StandardScaler`, fitted on the training set only to avoid **data leakage**.

### 4. Modeling & Evaluation
Trained a **Logistic Regression** model, chosen for its interpretability. Because the data is imbalanced (26.5% churn), accuracy alone is misleading: a model that predicts "no one churns" already scores 73.5%. The key metric is **Recall**, since missing a churning customer costs more than offering a discount to one who would have stayed.

## Key Findings

| Factor | Highest churn | Lowest churn |
|---|---|---|
| Contract type | Month-to-month: **42.7%** | Two year: **2.8%** |
| Tenure | 0–12 months: **47%** | 49–72 months: **9.5%** |
| Internet service | Fiber optic: **41.9%** | No internet: **7.4%** |
| Payment method | Electronic check: **45.3%** | Credit card (automatic): **15.2%** |
| Security & support services | Without: **~40%** | With: **15–22%** |

Streaming services showed almost no difference in churn (3–4 points).

![Tenure Distribution by Churn](<img width="576" height="455" alt="churn-tenure" src="https://github.com/user-attachments/assets/7f89d6f4-37b7-453b-a50d-afbbec5555a6" />
)

## Model Results

| Metric (churn class) | Baseline | Logistic Regression | LR (balanced) |
|---|---|---|---|
| Accuracy | 73.5% | 81% | 74% |
| Recall | 0% | 57% | **79%** |
| Precision | – | 66% | 51% |
| F1-score | – | 0.61 | 0.62 |

**Selected model:** Logistic Regression with `class_weight='balanced'`, which identifies **79% of churning customers** before they leave.

The model's strongest factors matched the EDA findings: longer tenure and two-year contracts are linked to lower churn, while month-to-month contracts and Fiber optic service are linked to higher churn.

## Recommendations

| Finding | Suggested action | How to measure |
|---|---|---|
| First-year customers churn at 47% | Onboarding and follow-up program during the first 12 months | Compare new-customer churn before and after |
| Month-to-month contracts churn at 42.7% | Discount or free month for upgrading to an annual contract | Churn of upgraders vs. non-upgraders |
| Fiber optic customers churn at 41.9% | Investigate the cause first (exit surveys, price and quality comparison with competitors) | Identify the main driver before acting |
| Customers without security/support churn at ~40% | Free trial of tech support and security for high-risk customers | Compare against a group without the offer |
| Electronic check customers churn at 45.3% | Small discount for switching to automatic payment | Churn of switchers vs. non-switchers |

**Using the model in practice:** score current customers monthly and target retention offers at those with the highest churn probability.

## Limitations

- Findings are **correlations, not causation**, so recommendations are framed as experiments to be measured.
- Some features overlap strongly (**multicollinearity**), e.g. `MonthlyCharges` with service columns and `TotalCharges` with `tenure`, so their individual model weights are not interpreted on their own.
- With 51% precision, about half of the alerts are false alarms. This is acceptable for low-cost offers, but should be revisited if retention offers are expensive.
- The dataset does not include direct reasons for leaving, such as satisfaction or service quality.

## How to Run

Click the **Open in Colab** badge at the top, then choose **Runtime → Run all**. The dataset is loaded directly from its source, so no setup is needed.

## Author

**Wejdan** · Information Systems graduate interested in data science, AI, and data engineering.

# 🏦 Credit Risk Default Prediction

## 📌 Project Overview
This project applies machine learning to predict credit loan defaults. By analyzing financial indicators and borrower profiles, the goal is to build a classification model that accurately identifies high-risk loans, ultimately saving financial institutions from costly write-offs while minimizing the rejection of good customers.

## 💼 The Business Problem
In credit risk assessment, a model makes two types of errors, both with significant financial consequences:
1. **False Negatives (Missing a Defaulter):** Approving a loan that eventually defaults. This results in the loss of the principal amount and expensive recovery efforts. This is the most costly error.
2. **False Positives (Flagging a Good Customer):** Denying a loan to a reliable customer, resulting in lost interest revenue and damaged customer trust.

**Metric of Choice:** Because the dataset is highly imbalanced (defaults make up only ~22% of the data), standard accuracy is highly misleading. This project optimizes for the **F1-Score** (to balance precision and recall) and **ROC-AUC** (to measure risk ranking capability).

## 📊 Exploratory Data Analysis (EDA) Highlights
Through visual exploration, several key patterns emerged that signal default risk:
* **Loan Grade:** As internal loan grades drop from A to G, the default rate drastically increases, proving the baseline grading system works but can be enhanced.
* **Interest Rates:** Defaulters heavily cluster around higher interest rates (>13%).
* **Income-to-Loan Ratio:** Borrowers taking loans that consume a massive percentage of their annual income are the most likely to default.

## 🤖 Modeling Strategy
The project follows a progressive approach, starting from simple, highly-interpretable models and moving to complex ensembles:

1. **Baseline Decision Tree:** Highly overfit the training data (memorization).
2. **Regularized Decision Tree:** Applied manual constraints (`max_depth`, `min_samples_split`) to reduce variance.
3. **Tuned Decision Tree:** Utilized **Optuna** for automated hyperparameter tuning.
4. **Random Forest:** Introduced bagging to drastically reduce variance.
5. **XGBoost (Winner):** Gradient boosting effectively minimized both bias and variance, resulting in the strongest predictive performance.

### 🏆 Final Model Performance (XGBoost)
* **F1-Score:** ~0.83
* **ROC-AUC:** ~0.94

## 🔍 Feature Importance & Insights
By extracting the feature importance from the XGBoost model, the top drivers of default risk are:
1. **Loan Percent Income:** The ratio of the loan amount to the borrower's income.
2. **Loan Grade:** The institution's assigned risk category.
3. **Interest Rate:** The assigned interest rate of the loan.

*Note: Following feature selection analysis (Elbow Curve), bottom-performing, "noisy" features can be successfully dropped without degrading the model's F1-Score, resulting in a faster, more interpretable model.*

## 🛠️ Tools & Libraries Used
* **Python 3**
* **Pandas & NumPy:** Data manipulation
* **Scikit-Learn:** Decision Trees, Random Forests, Metrics, Preprocessing
* **XGBoost:** Gradient boosting classification
* **Optuna:** Advanced hyperparameter optimization
* **Plotly & Seaborn:** Interactive and static data visualization

## 🚀 How to Run this Project
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/FattyGarfield/Credit-Risk-Prediction.git](https://github.com/FattyGarfield/Credit-Risk-Prediction.git)

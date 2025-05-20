# Churn Prediction 📉 (Note: Project still in progress )

This repository contains a Jupyter Notebook for predicting customer churn. The goal is to identify customers who are likely to cancel their contracts soon, enabling the company to take proactive measures to retain them. This is a binary classification problem, where the target variable is "churn" (yes/no).

The dataset used in this project is sourced from Kaggle: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn).

## Dataset Description 🗃️

The dataset includes information about customer services, account details, charges, and demographic information.
* **Services of the customers**: phone; multiple lines; internet; tech support and extra services such as online security, backup, device protection, and TV streaming
* **Account information**: how long they have been clients, type of contract, type of payment method
* **Charges**: how much the client was charged in the past month and in total
* **Demographic information**: gender, age, and whether they have dependents or a partner
* **Churn**: yes/no, whether the customer left the company within the past month

## Initial Data Preparation 🛠️

Several steps were taken to prepare the data for analysis and modeling:

* **Customer ID Handling**: The `customerID` column was dropped as it is not an important feature for prediction.
* **`TotalCharges` Preprocessing**:
    * The `TotalCharges` column had white spaces in 11 rows, which needed to be addressed.
    * The `TotalCharges` column was converted to a numeric type, with white spaces enforced to be `NaN`.
* **Missing Value Imputation**: The missing values in `TotalCharges` were handled by imputing them with the median, as the distribution of `TotalCharges` is skewed.
* **Churn Imbalance**: It was observed that the `Churn` column had an imbalance in its distribution.

## Exploratory Data Analysis (EDA) 📊

EDA was performed to understand the relationships within the data:

* **Data Splitting**: The data was split into categorical and numerical features for easier analysis.
* **Categorical-Target Correlation**: Correlation between the target (`Churn`) and categorical columns was checked by grouping by the mean of `Churn` and comparing with each categorical column.
    * **Insights from Categorical Analysis**:
        * Gender showed little difference in churn rates between females and males.
        * Senior citizens tended to churn more than non-seniors.
        * Customers with a partner churned less than those without a partner.
        * People using phone service were not at risk of churning, and those not using it were even less likely to churn.
        * Clients without tech support tended to churn more than those who had it.
        * Customers with monthly contracts canceled more often, while those with two-year contracts churned very rarely.
* **Mutual Information (MI) Score**: Higher mutual information between a categorical variable and the target indicates a stronger dependence and thus better predictive utility. Low mutual information suggests independence and less usefulness for prediction.
* **Numerical-Target Correlation**:
    * **Tenure**: The correlation between `tenure` and `churn` was -0.35, indicating that longer-staying customers churn less often.
    * **Monthly Charges**: `monthlycharges` had a positive coefficient of 0.19, meaning customers paying more tended to leave more often.
    * **Total Charges**: `totalcharges` had a negative correlation, which makes sense as customers who stay longer and pay more in total are less likely to leave.

## Data Splitting ➖

The dataset was split into training and testing sets to evaluate the model's performance on unseen data.

## Preprocessing ⚙️

Further preprocessing steps were applied before modeling:

* **Scaling Numerical Features**: A gap was observed between the mean of numerical values, necessitating scaling. `StandardScaler` was used for this purpose.
* **Handling Categorical Variables**: One-hot encoding was used to convert categorical variables into a numerical format suitable for machine learning models.

## Modeling 🧠

Logistic Regression was chosen as the classification model for churn prediction.

## Training and Evaluation After EDA 📈

After conducting EDA, the training and evaluation process was repeated:

* **Handling Imbalanced Churn Data**: The imbalanced `Churn` data needed to be addressed.
* **Oversampling**: Oversampling was applied to generate synthetic compositions to balance the dataset's distribution.
* **Feature Selection**: The most important categorical features were kept based on their high Mutual Information (MI) Score.

# 🚨 Financial Fraud Detection using Python & Pandas

## 📌 Project Overview

This project analyzes over **1 Million Financial Transactions** to identify fraud patterns, transaction behavior, and risk indicators using Python and Pandas.

The project follows a complete Data Analytics workflow:

**Data Loading → Data Cleaning → Feature Engineering → Data Aggregation → Fraud Analysis → Insights Generation**

---


## 🎯 Objectives

* Analyze large-scale financial transaction data.
* Identify fraudulent transactions.
* Discover high-risk transaction types.
* Generate actionable business insights.
* Perform exploratory data analysis using Pandas.

---

## 📂 Dataset Information
Raw Datasets -https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset

**Dataset Size:** 1M+ Records

### Features

| Column         | Description                         |
| -------------- | ----------------------------------- |
| step           | Time unit of transaction            |
| type           | Transaction type                    |
| amount         | Transaction amount                  |
| nameOrig       | Sender account                      |
| oldbalanceOrg  | Sender balance before transaction   |
| newbalanceOrig | Sender balance after transaction    |
| nameDest       | Receiver account                    |
| oldbalanceDest | Receiver balance before transaction |
| newbalanceDest | Receiver balance after transaction  |
| isFraud        | Fraud indicator (0 = No, 1 = Yes)   |
| isFlaggedFraud | Flagged fraud transaction           |

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook / VS Code

---

## 🔄 Project Workflow

### 1️⃣ Data Loading

* Imported dataset using Pandas.
* Verified dataset structure and metadata.

### 2️⃣ Data Cleaning

* Checked missing values.
* Removed duplicate records.
* Validated data types.
* Converted transaction amount into numeric format.

### 3️⃣ Feature Engineering

Created:

* Amount Categories

  * Low
  * Medium
  * High

### 4️⃣ Data Aggregation

Performed:

* GroupBy Analysis
* Pivot Table Analysis
* Summary Statistics
* Fraud Rate Calculation

### 5️⃣ Fraud Detection Analysis

Analyzed:

* Total Fraud Transactions
* Fraud Percentage
* Fraud by Transaction Type
* High-Value Fraud Transactions
* Step-wise Fraud Trends

---

## 📊 Business Questions Solved

### Intermediate Analysis

✔ Total Transactions

✔ Total Transaction Amount

✔ Average Transaction Amount

✔ Transaction Type Distribution

✔ Fraud Transaction Count

✔ Fraud Percentage

✔ Transaction Summary Report

✔ Amount Analysis by Transaction Type

✔ Top 10 Highest Transactions

✔ Fraud Distribution Analysis

### Advanced Analysis

✔ Top Customers by Transaction Amount

✔ Fraud Rate by Transaction Type

✔ Top 1% High-Value Transactions

✔ Transaction Type Contribution Analysis

✔ Step-wise Fraud Trend Analysis

---


## 📊 Sample Output

Transaction Summary:

```python
type      total_amount      avg_amount     fraud_count
CASH_IN      XXXXX             XXXXX            XX
CASH_OUT     XXXXX             XXXXX            XX
PAYMENT      XXXXX             XXXXX            XX
TRANSFER     XXXXX             XXXXX            XX
```

# Financial Fraud Detection Project – Questions, Solutions & Insights

## Q1. What is the total number of transactions?

Solution:

```python
print(len(df))
```

Insight: Shows the overall dataset size and transaction volume.

## Q2. What is the total transaction amount?

Solution:

```python
print(df['amount'].sum())
```

Insight: Represents the total money transferred in the system.

## Q3. What is the average transaction amount?

Solution:

```python
print(df['amount'].mean())
```

Insight: Helps identify the typical transaction size.

## Q4. What is the maximum transaction amount?

Solution:

```python
print(df['amount'].max())
```

Insight: Identifies the largest transaction in the dataset.

## Q5. What is the minimum transaction amount?

Solution:

```python
print(df['amount'].min())
```

Insight: Identifies the smallest transaction value.

## Q6. How many unique transaction types exist?

Solution:

```python
print(df['type'].nunique())
```

Insight: Measures transaction diversity.

## Q7. What are the unique transaction types?

Solution:

```python
print(df['type'].unique())
```

Insight: Lists all available transaction categories.

## Q8. How many transactions belong to each type?

Solution:

```python
print(df['type'].value_counts())
```

Insight: Reveals the most common transaction type.

## Q9. What is the total amount by transaction type?

Solution:

```python
print(df.groupby('type')['amount'].sum())
```

Insight: Shows which transaction type handles the most money.

## Q10. What is the average amount by transaction type?

Solution:

```python
print(df.groupby('type')['amount'].mean())
```

Insight: Compares transaction size across categories.

## Q11. What is the maximum amount by transaction type?

Solution:

```python
print(df.groupby('type')['amount'].max())
```

Insight: Identifies the largest transaction in each category.

## Q12. What is the minimum amount by transaction type?

Solution:

```python
print(df.groupby('type')['amount'].min())
```

Insight: Identifies the smallest transaction in each category.

## Q13. What is the median transaction amount by type?

Solution:

```python
print(df.groupby('type')['amount'].median())
```

Insight: Provides a better central tendency measure than mean.

## Q14. What is the standard deviation by transaction type?

Solution:

```python
print(df.groupby('type')['amount'].std())
```

Insight: Measures transaction variability.

## Q15. How many fraud transactions exist?

Solution:

```python
print(df['isFraud'].sum())
```

Insight: Core fraud metric of the project.

## Q16. What is the fraud percentage?

Solution:

```python
fraud_rate = (df['isFraud'].sum()/len(df))*100
print(fraud_rate)
```

Insight: Indicates fraud prevalence in the dataset.

## Q17. How many fraud transactions occur by type?

Solution:

```python
print(df.groupby('type')['isFraud'].sum())
```

Insight: Identifies high-risk transaction categories.

## Q18. What are the top 10 highest transactions?

Solution:

```python
print(df.nlargest(10,'amount'))
```

Insight: Detects unusually large transactions.

## Q19. Which transactions are above average amount?

Solution:

```python
avg_amount = df['amount'].mean()
print(df[df['amount'] > avg_amount])
```

Insight: Highlights significant transactions.

## Q20. Generate a transaction summary report.

Solution:

```python
summary = df.groupby('type').agg(
    total_amount=('amount','sum'),
    avg_amount=('amount','mean'),
    count_transactions=('amount','count'),
    fraud_count=('isFraud','sum')
)
print(summary)
```

Insight: Provides a complete business overview.

# Advanced Questions

## ADV Q1. Who are the top 5 customers by transaction amount?

Solution:

```python
top_customers = (
    df.groupby('nameOrig')['amount']
      .sum()
      .sort_values(ascending=False)
      .head(5)
)
print(top_customers)
```

Insight: Identifies high-value customers.

## ADV Q2. What is the fraud rate by transaction type?

Solution:

```python
fraud_rate = (
    df.groupby('type')
      .agg(
          total_transactions=('isFraud','count'),
          fraud_transactions=('isFraud','sum')
      )
)

fraud_rate['fraud_rate_%'] = (
    fraud_rate['fraud_transactions']
    / fraud_rate['total_transactions']
) * 100

print(fraud_rate)
```

Insight: Reveals the riskiest transaction types.

## ADV Q3. Which transactions belong to the top 1% by amount?

Solution:

```python
threshold = df['amount'].quantile(0.99)

high_value_transactions = df[
    df['amount'] >= threshold
]

print(high_value_transactions)
```

Insight: Detects unusual high-value transactions.

## ADV Q4. What is the contribution percentage of each transaction type?

Solution:

```python
contribution = (
    df.groupby('type')['amount']
      .sum()
      .reset_index()
)

contribution['contribution_%'] = (
    contribution['amount']
    / contribution['amount'].sum()
) * 100

print(contribution)
```

Insight: Shows which transaction types drive the most business value.

## ADV Q5. What is the step-wise fraud trend?

Solution:

```python
step_analysis = (
    df.groupby('step')
      .agg(
          total_transactions=('isFraud','count'),
          fraud_transactions=('isFraud','sum'),
          total_amount=('amount','sum')
      )
)

step_analysis['fraud_rate_%'] = (
    step_analysis['fraud_transactions']
    / step_analysis['total_transactions']
) * 100

print(step_analysis)
```

Insight: Tracks fraud behavior over time and identifies risky periods.

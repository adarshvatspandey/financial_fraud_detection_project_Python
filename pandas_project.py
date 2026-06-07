#project : capstone project of financial data usind python
# loading -> cleaning -> transformation -> aggregation -> visualization

# Step1: import libs
import pandas as pd 
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns

#step2: load the data
df = pd.read_csv("D:/Financial fraud detection python datasets/archive/Synthetic_Financial_datasets_log.csv")
print(df.head()) #output 5 rows data
print(df.info())  #check metadata

# step3: data cleaning

# handle missing values , correct  data types , removes duplicate

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print(df.info())



#STEP4: transformation :

# feature engineering new column addition

# categorization - group of numeric data for better analysis.
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['amount_category'] = pd.cut(
    df['amount'],
    bins=[0, 5000, 10000, 20000],
    labels=['low', 'medium', 'high']
)

print(df.head(5))

print(df[['amount']].head(10))
print(df['amount'].dtype)

# step5: data aggregation - groupby or pivot using pandas

transaction_summary = df.groupby("type").agg(
    total_amount=("amount", "sum"),
    avg_amount=("amount", "mean"),
    count_trans=("amount", "count")
).reset_index()

print(transaction_summary)

# ===========================
# 20 INTERMEDIATE PANDAS QUESTIONS & SOLUTIONS
# ===========================

# Q1. Top 5 highest transaction amounts
print("\nQ1. Top 5 Highest Transactions")
print(df.nlargest(5, 'amount'))

# Q2. Top 5 lowest transaction amounts
print("\nQ2. Top 5 Lowest Transactions")
print(df.nsmallest(5, 'amount'))

# Q3. Unique transaction types
print("\nQ3. Unique Transaction Types")
print(df['type'].unique())

# Q4. Number of unique transaction types
print("\nQ4. Number of Unique Transaction Types")
print(df['type'].nunique())

# Q5. Transaction type frequency
print("\nQ5. Transaction Type Frequency")
print(df['type'].value_counts())

# Q6. Percentage distribution of transaction types
print("\nQ6. Percentage Distribution of Transaction Types")
print(df['type'].value_counts(normalize=True) * 100)

# Q7. Average amount by transaction type
print("\nQ7. Average Amount by Transaction Type")
print(df.groupby('type')['amount'].mean())

# Q8. Maximum amount by transaction type
print("\nQ8. Maximum Amount by Transaction Type")
print(df.groupby('type')['amount'].max())

# Q9. Minimum amount by transaction type
print("\nQ9. Minimum Amount by Transaction Type")
print(df.groupby('type')['amount'].min())

# Q10. Median amount by transaction type
print("\nQ10. Median Amount by Transaction Type")
print(df.groupby('type')['amount'].median())

# Q11. Standard deviation by transaction type
print("\nQ11. Standard Deviation by Transaction Type")
print(df.groupby('type')['amount'].std())

# Q12. Total fraud transactions by type
print("\nQ12. Total Fraud Transactions by Type")
print(df.groupby('type')['isFraud'].sum())

# Q13. Fraud rate by transaction type
print("\nQ13. Fraud Rate (%) by Transaction Type")
print(df.groupby('type')['isFraud'].mean() * 100)

# Q14. Top 10 fraud transactions
print("\nQ14. Top 10 Fraud Transactions")
print(df[df['isFraud'] == 1].nlargest(10, 'amount'))

# Q15. Transactions above average amount
print("\nQ15. Transactions Above Average Amount")
avg_amount = df['amount'].mean()
print(df[df['amount'] > avg_amount])

# Q16. Sort transactions by amount descending
print("\nQ16. Transactions Sorted by Amount (Descending)")
print(df.sort_values('amount', ascending=False))

# Q17. Multi-column aggregation
print("\nQ17. Multi-Column Aggregation")
print(
    df.groupby('type').agg({
        'amount': ['sum', 'mean', 'max'],
        'isFraud': 'sum'
    })
)

# Q18. Pivot Table Analysis
print("\nQ18. Pivot Table Analysis")
print(
    pd.pivot_table(
        df,
        values='amount',
        index='type',
        columns='isFraud',
        aggfunc='sum'
    )
)

# Q19. Correlation between numeric columns
print("\nQ19. Correlation Matrix")
print(df.corr(numeric_only=True))

# Q20. Transaction Summary Report
print("\nQ20. Transaction Summary Report")
summary = df.groupby('type').agg(
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
    max_amount=('amount', 'max'),
    fraud_count=('isFraud', 'sum')
)
print(summary)

# ==================================
# ADVANCED Q1. Top 5 Customers by Total Transaction Amount
# ==================================
top_customers = (
    df.groupby('nameOrig')['amount']
      .sum()
      .sort_values(ascending=False)
      .head(5)
)

print("\nADV Q1. Top 5 Customers by Total Amount")
print(top_customers)


# ==================================
# ADVANCED Q2. Fraud Rate by Transaction Type
# ==================================
fraud_rate = (
    df.groupby('type')
      .agg(
          total_transactions=('isFraud', 'count'),
          fraud_transactions=('isFraud', 'sum')
      )
)

fraud_rate['fraud_rate_%'] = (
    fraud_rate['fraud_transactions']
    / fraud_rate['total_transactions']
) * 100

print("\nADV Q2. Fraud Rate by Transaction Type")
print(fraud_rate)


# ==================================
# ADVANCED Q3. Detect High Value Transactions (Top 1%)
# ==================================
threshold = df['amount'].quantile(0.99)

high_value_txn = df[df['amount'] >= threshold]

print("\nADV Q3. Top 1% High Value Transactions")
print(high_value_txn[['type', 'amount', 'isFraud']].head())


# ==================================
# ADVANCED Q4. Transaction Type Contribution %
# ==================================
contribution = (
    df.groupby('type')['amount']
      .sum()
      .reset_index()
)

contribution['contribution_%'] = (
    contribution['amount']
    / contribution['amount'].sum()
) * 100

print("\nADV Q4. Contribution % by Transaction Type")
print(contribution)


# ==================================
# ADVANCED Q5. Step-wise Fraud Analysis
# ==================================
step_fraud = (
    df.groupby('step')
      .agg(
          total_transactions=('isFraud', 'count'),
          fraud_transactions=('isFraud', 'sum'),
          total_amount=('amount', 'sum')
      )
)

step_fraud['fraud_rate_%'] = (
    step_fraud['fraud_transactions']
    / step_fraud['total_transactions']
) * 100

print("\nADV Q5. Step-wise Fraud Analysis")
print(step_fraud.head(20))
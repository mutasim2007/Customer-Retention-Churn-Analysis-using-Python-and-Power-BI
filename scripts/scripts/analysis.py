# =============================
# Customer Retention & Churn Analysis
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Sample Dataset
# -----------------------------
np.random.seed(42)

n = 500

data = pd.DataFrame({
    'CustomerID': range(1, n + 1),
    'Plan': np.random.choice(['Basic', 'Standard', 'Premium'], n),
    'Region': np.random.choice(['North America', 'Europe', 'Asia Pacific', 'Middle East', 'Latin America'], n),
    'Tenure_Months': np.random.randint(1, 24, n),
    'Engagement_Score': np.random.randint(10, 100, n),
    'Monthly_Spend': np.random.randint(20, 200, n),
    'Churn': np.random.choice([0, 1], n, p=[0.7, 0.3])
})

# Add churn reasons
reasons = ['Price too high', 'Poor support', 'Found better alternative', 'Low usage', 'Technical issues']
data['Churn_Reason'] = np.where(data['Churn'] == 1,
                                np.random.choice(reasons, n),
                                'Active')

# Save dataset
data.to_csv('customer_data.csv', index=False)

print("Dataset created successfully!\n")
print(data.head())

# -----------------------------
# 2. Churn Rate by Plan
# -----------------------------
churn_by_plan = data.groupby('Plan')['Churn'].mean() * 100
print("\nChurn Rate by Plan:\n", churn_by_plan)

# -----------------------------
# 3. Retention Rate by Region
# -----------------------------
retention_by_region = (1 - data.groupby('Region')['Churn'].mean()) * 100
print("\nRetention Rate by Region:\n", retention_by_region)

# -----------------------------
# 4. Customer Lifetime Value (CLV)
# -----------------------------
data['CLV'] = data['Monthly_Spend'] * data['Tenure_Months']
clv_by_plan = data.groupby('Plan')['CLV'].mean()
print("\nAverage CLV by Plan:\n", clv_by_plan)

# -----------------------------
# 5. Monthly Churn Trend
# -----------------------------
data['Month'] = np.random.choice(
    pd.date_range('2022-01-01', '2024-12-01', freq='MS'), n
)

churn_trend = data.groupby('Month')['Churn'].sum()

# -----------------------------
# 6. Cohort Analysis
# -----------------------------
data['Cohort'] = data['Month'].dt.to_period('M')
cohort_table = data.groupby(['Cohort', 'Tenure_Months']).size().unstack(fill_value=0)

# -----------------------------
# 7. Visualizations
# -----------------------------

# Churn Rate by Plan
plt.figure()
churn_by_plan.plot(kind='bar', title='Churn Rate by Plan')
plt.xlabel('Plan')
plt.ylabel('Churn Rate (%)')
plt.show()

# Retention Rate by Region
plt.figure()
retention_by_region.plot(kind='bar', title='Retention Rate by Region')
plt.xlabel('Region')
plt.ylabel('Retention Rate (%)')
plt.show()

# Customer Lifetime Value
plt.figure()
clv_by_plan.plot(kind='bar', title='Customer Lifetime Value by Plan')
plt.xlabel('Plan')
plt.ylabel('Average CLV')
plt.show()

# Monthly Churn Trend
plt.figure()
churn_trend.plot(title='Monthly Churn Trend')
plt.xlabel('Month')
plt.ylabel('Churn Count')
plt.show()

print("\nAnalysis Completed Successfully!")

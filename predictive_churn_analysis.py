
# Predictive Churn Analysis
# Customer Churn Prediction using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Create Dataset
data = {
    'CustomerID': range(1001, 1021),
    'Gender': ['Male', 'Female'] * 10,
    'Age': [25, 30, 45, 35, 50, 28, 40, 32, 60, 29,
            55, 38, 42, 26, 48, 33, 52, 31, 44, 27],
    'Tenure': [12, 24, 5, 36, 2, 18, 8, 30, 1, 20,
               4, 28, 10, 15, 3, 32, 6, 22, 9, 16],
    'MonthlyCharges': [500, 700, 900, 600, 1000, 550, 850,
                       650, 1100, 500, 950, 750, 800, 520,
                       980, 620, 1050, 680, 870, 530],
    'Contract': ['Monthly', 'Yearly', 'Monthly', 'Yearly',
                 'Monthly', 'Yearly', 'Monthly', 'Yearly',
                 'Monthly', 'Yearly', 'Monthly', 'Yearly',
                 'Monthly', 'Yearly', 'Monthly', 'Yearly',
                 'Monthly', 'Yearly', 'Monthly', 'Yearly'],
    'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes',
              'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No',
              'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 2. Display Dataset
print("CUSTOMER DATA")
print(df.head(20))

# 3. Data Information
print("\nDATA INFORMATION")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

# 4. Data Visualization
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn')
plt.title("Customer Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title("Churn by Contract Type")
plt.savefig("churn_by_contract.png")
plt.show()

# 5. Data Preprocessing
encoder = LabelEncoder()

df['Gender'] = encoder.fit_transform(df['Gender'])
df['Contract'] = encoder.fit_transform(df['Contract'])
df['Churn'] = encoder.fit_transform(df['Churn'])

# 6. Select Features and Target
X = df[['Gender', 'Age', 'Tenure', 'MonthlyCharges', 'Contract']]
y = df['Churn']

# 7. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 8. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Machine Learning Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 10. Make Predictions
y_pred = model.predict(X_test)

# 11. Model Evaluation
print("\nMODEL ACCURACY")
print(accuracy_score(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nPREDICTIVE CHURN ANALYSIS COMPLETED!")

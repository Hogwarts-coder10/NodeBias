import pandas as pd
import numpy as np
import joblib

from glassboxml.core import train_test_split, Momentum
from glassboxml.preprocessing import StandardScaler
from glassboxml.models import GaussianNaiveBayes,LogisticRegression

print("🔧 Welding models into memory...")

# 1. Load and clean the baseline data
df = pd.read_csv('diabetic_data.csv')
df.replace('?', np.nan, inplace=True)
df = df[df['gender'].isin(['Male', 'Female'])]
df['is_readmitted'] = df['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)
df['gender_Male'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

safe_features = ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_diagnoses']
df.dropna(subset=safe_features + ['gender_Male'], inplace=True)

X = df[safe_features].values.astype(float)
y = df['is_readmitted'].values.astype(float)

# 2. Scale the data and FREEZE the Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'nodebias_scaler.pkl')
print("✅ Scaler frozen as 'nodebias_scaler.pkl'")

# 3. Train and FREEZE Custom Logistic Regression
optimizer = Momentum(learning_rate = 0.01,beta = 0.9)
log_reg = LogisticRegression(optimizer=optimizer,epochs=10000,loss_function = 'bce')
log_reg.fit(X_scaled, y)
joblib.dump(log_reg, 'nodebias_logreg.pkl')
print("✅ Logistic Regression engine frozen as 'nodebias_logreg.pkl'")

# 4. Train and FREEZE Custom Naive Bayes
gnb = GaussianNaiveBayes()
gnb.fit(X_scaled, y)
joblib.dump(gnb, 'nodebias_gnb.pkl')
print("✅ Naive Bayes engine frozen as 'nodebias_gnb.pkl'")

print("🚀 Welding complete. Chassis is secure.")

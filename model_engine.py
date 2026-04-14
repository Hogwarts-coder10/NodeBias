import pandas as pd
import numpy as np
import json

from glassboxml.core import train_test_split, Momentum
from glassboxml.models import LogisticRegression
from glassboxml.preprocessing import StandardScaler
from data_pipeline import clean_medical_data

print("1. Loading and Cleaning Data...")
df = pd.read_csv('diabetic_data.csv')

df['is_readmitted'] = df['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)
# Pull sanitized tensors from the pipeline (is_mitigated=False leaves gender in the dataset)
X, y, raw_genders, features, _ = clean_medical_data(df, is_mitigated=False)

print("2. Splitting Data (Preventing Leakage)...")
X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, raw_genders, test_size=0.2, random_state=42
)

print("3. Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("4. Downsampling the Training Set...")
train_df = pd.DataFrame(X_train_scaled, columns=features)
train_df['target'] = y_train

class_0 = train_df[train_df['target'] == 0]
class_1 = train_df[train_df['target'] == 1]

class_0_downsampled = class_0.sample(n=len(class_1), random_state=42)
balanced_train_df = pd.concat([class_0_downsampled, class_1]).sample(frac=1, random_state=42)

X_train_balanced = balanced_train_df[features].values
y_train_balanced = balanced_train_df['target'].values

print("5. Training Custom GlassBoxML Model...")
optimizer = Momentum(learning_rate=0.01, beta=0.9)
model = LogisticRegression(optimizer=optimizer, epochs=1000, loss_function='bce')
model.fit(X_train_balanced, y_train_balanced)

print("6. Generating Predictions on Unseen Test Data...")
predictions = model.predict(X_test_scaled)

results_df = pd.DataFrame({'gender': gender_test, 'predicted_readmission': predictions})

print("7. Auditing Model & Extracting Explanations...")
model_bias_report = {}
for group in ['Male', 'Female']:
    group_data = results_df[results_df['gender'] == group]
    predicted_rate = group_data['predicted_readmission'].mean()

    model_bias_report[group] = {
        "total_tested": len(group_data),
        "ai_predicted_readmission_rate_percentage": round(predicted_rate * 100, 2)
    }

male_rate = model_bias_report["Male"]["ai_predicted_readmission_rate_percentage"]
female_rate = model_bias_report["Female"]["ai_predicted_readmission_rate_percentage"]
disparate_impact = round((male_rate / female_rate), 3) if female_rate > 0 else "N/A"

model_bias_report["Metrics"] = {
    "Disparate_Impact_Ratio": disparate_impact,
    "Status": "Biased" if type(disparate_impact) != str and disparate_impact < 0.8 else "Fair"
}

model_bias_report["Model_Explanation"] = model.explain()

with open('model_bias_report.json', 'w') as f:
    json.dump(model_bias_report, f, indent=4)

print("✅ Logistic Regression Backend Complete. JSON Generated.")

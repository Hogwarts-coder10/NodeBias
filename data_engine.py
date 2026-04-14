import pandas as pd
import json
from data_pipeline import clean_medical_data

print("Loading Dataset.....")
df = pd.read_csv("diabetic_data.csv")

df['is_readmitted'] = df['readmitted'].apply(lambda x: 1 if x in ['<30', '>30'] else 0)
# Use the pipeline to get the sanitized dataframe
_, _, _, _, cleaned_df = clean_medical_data(df, is_mitigated=False)

sensitive_col = 'gender'
print(f"Auditing raw dataset for {sensitive_col} bias...")

bias_report = {}
groups = cleaned_df[sensitive_col].unique()

for group in groups:
    group_data = cleaned_df[cleaned_df[sensitive_col] == group]
    readmission_rate = group_data['is_readmitted'].mean()

    bias_report[group] = {
        "total_patients": len(group_data),
        "readmission_rate_percentage": round(readmission_rate * 100, 2)
    }

output_file = 'data_bias_report.json'
with open(output_file, 'w') as f:
    json.dump(bias_report, f, indent=4)

print(f"✅ Phase 1 Complete! Data Bias JSON saved to {output_file}.")
print(json.dumps(bias_report, indent=4))

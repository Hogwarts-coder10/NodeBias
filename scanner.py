import re
import pandas as pd

def universal_sensitive_scanner(df, sample_size=50):
    """
    Universally scans a DataFrame for protected attributes by checking
    both the column headers and a sample of the underlying data.
    """
    detected_cols = []

    # 1. EXPANDED DICTIONARY (Universal Protected Classes & Proxies)
    header_patterns = {
        'gender_sex': r'\b(gender|sex|sexual_orientation)\b',
        'race_ethnicity': r'\b(race|ethnicity|origin|caste|nationality)\b',
        'age': r'\b(age|dob|date_of_birth|birth_year)\b',
        'religion': r'\b(religion|belief|faith)\b',
        'health_disability': r'\b(disability|handicap|medical|pregnant)\b',
        'financial_status': r'\b(income|salary|wealth|credit_score)\b',
        'family_status': r'\b(marital|spouse|children|dependents)\b',
        'geographic_proxy': r'\b(zip|pincode|neighborhood|postcode)\b' # Often used for redlining
    }

    # 2. VALUE HEURISTICS (For when headers are vague like "Attr_1")
    value_keywords = {
        'gender_sex': ['male', 'female', 'non-binary', 'transgender'],
        'race_ethnicity': ['caucasian', 'asian', 'hispanic', 'latino', 'black', 'african american', 'white']
    }

    # Grab a small sample of data to check values safely
    sample_df = df.head(sample_size).astype(str).apply(lambda x: x.str.lower())

    for col in df.columns:
        col_lower = str(col).lower().replace('_', ' ')
        flagged = False

        # Phase A: Check the Header Name
        for category, pattern in header_patterns.items():
            if re.search(pattern, col_lower):
                detected_cols.append({"column": col, "category": category, "method": "header_match"})
                flagged = True
                break

        # Phase B: Check the Values (If header check missed)
        if not flagged:
            unique_vals = sample_df[col].unique()
            for val in unique_vals:
                for category, keywords in value_keywords.items():
                    if any(kw in str(val) for kw in keywords):
                        detected_cols.append({"column": col, "category": category, "method": "value_inference"})
                        flagged = True
                        break
                if flagged: break # Move to next column if found

    return detected_cols

import pandas as pd

def generate_feature_engineering_prompt(dataset_description: str, df: pd.DataFrame, target_col: str = None) -> str:
    """
    Membuat prompt terstruktur untuk LLM Agent dengan perlindungan Data Leakage.
    """
    # 1. Ambil Schema
    schema_info = df.dtypes.to_string()
    
    # 2. Ambil Statistik Singkat (Agar LLM tahu distribusi data, bukan cuma sampel)
    # Ini membantu LLM tahu ada nilai negatif atau tidak, range nilai, dll.
    stats_info = df.describe().to_string()
    
    # 3. Ambil Sample CSV
    csv_sample = df.head(5).to_csv(index=False)
    
    # 4. Tentukan Pesan Target (Penting untuk konteks prediksi)
    target_context = ""
    if target_col:
        target_context = f"The TARGET variable we want to predict is: '{target_col}'."
    
    prompt = f"""
### ROLE
You are an expert Data Scientist and Feature Engineering Agent. Your goal is to create new, highly predictive features from the provided dataset to improve the accuracy of a Machine Learning model.

### CONTEXT
1. **Dataset Description**: {dataset_description}
2. **Target Variable**: {target_context}
3. **Schema Info**:
{schema_info}
4. **Data Statistics**:
{stats_info}
5. **Data Sample (First 5 rows)**:
{csv_sample}

### INSTRUCTION
Analyze the dataset and suggest 5 high-impact new features that could help predict the target '{target_col}'.
For each feature, write a valid **Pandas Python Expression** that can be executed to create the feature.

### CONSTRAINTS (STRICT)
1. **Input Data**: The dataframe variable is named `df`.
2. **Libraries**: You can use `pd` (pandas) and `np` (numpy). Do NOT use `import`.
3. **NO DATA LEAKAGE**: You MUST NOT use the target column ('{target_col}') inside the 'expression'. The target is what we want to predict, it cannot be part of the input features.
4. **Robustness**: Handle "division by zero" using `+ 1e-6` or `np.where`. Handle log of negative numbers if necessary.
5. **Format**: Output MUST be a valid JSON list. Do not include markdown formatting (```json).

### OUTPUT FORMAT EXAMPLE
[
  {{
    "name": "Income_Per_Capita",
    "expression": "df['Total_Income'] / (df['Family_Size'] + 1e-6)",
    "rationale": "Normalizes income by family size to measure real purchasing power."
  }},
  {{
    "name": "Log_Transaction_Amount",
    "expression": "np.log1p(df['Transaction_Amount'])",
    "rationale": "Log transformation to handle skewed distribution in transaction amounts."
  }}
]

### YOUR RESPONSE (JSON ONLY):
"""
    return prompt
import pandas as pd

def generate_feature_engineering_prompt(dataset_description: str, df: pd.DataFrame) -> str:
    """
    Membuat prompt terstruktur untuk LLM Agent.
    Mengirimkan Metadata + Sampel Data untuk hasil yang akurat.
    """
    # Ambil Schema (Tipe Data)
    schema_info = df.dtypes.to_string()
    
    # Ambil Sample (5 baris) sebagai CSV string
    # index=False agar LLM tidak bingung dengan nomor baris
    csv_sample = df.head(5).to_csv(index=False)
    
    prompt = f"""
### ROLE
You are an expert Data Scientist and Feature Engineering Agent. Your goal is to create new, predictive features from the provided dataset to improve the accuracy of a Machine Learning model.

### CONTEXT
1. **Dataset Description**: {dataset_description}
2. **Schema Info**:
{schema_info}
3. **Data Sample (First 5 rows)**:
{csv_sample}

### INSTRUCTION
Analyze the dataset and suggest 3 high-impact new features.
For each feature, write a valid **Pandas Python Expression** that can be executed to create the feature.

### CONSTRAINTS (STRICT)
1. **Input Data**: The dataframe variable is named `df`.
2. **Safety**: Do NOT use `import`, `os`, `sys`, or any external libraries. Use only standard Pandas/Python math operators.
3. **Robustness**: Handle potential "division by zero" by adding a small epsilon (e.g., `+ 1e-6`) or using `.replace()`.
4. **Format**: Output MUST be a raw JSON list. Do not include markdown formatting (```json).

### OUTPUT FORMAT EXAMPLE
[
  {{
    "name": "Income_Per_Capita",
    "expression": "df['Total_Income'] / (df['Family_Size'] + 1e-6)",
    "rationale": "Normalizes income by family size to measure real purchasing power."
  }},
  {{
    "name": "Is_Weekend",
    "expression": "(pd.to_datetime(df['Transaction_Date']).dt.dayofweek >= 5).astype(int)",
    "rationale": "Captures weekly spending behavior patterns."
  }}
]

### YOUR RESPONSE (JSON ONLY):
"""
    return prompt
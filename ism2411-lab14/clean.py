"""
====================================================================
ISM2411 Lab 14 — Self-Assessment Reflection
====================================================================
1. Which of the three cleaning decisions — drop, fill, or flag — feels most 
   ambiguous to you? What additional information would you want before 
   making that call on real data?
   - Filling missing numeric values with the mean feels most ambiguous because 
     imputation introduces synthetic data that may distort variance or skew 
     distributions. Before filling, I would want to know the mechanism behind 
     the missingness and whether 
     domain experts can provide more accurate estimates or if dropping rows 
     is safer.

2. Look at your comments. If someone who didn't know the dataset read your 
   script, would they understand why you made each decision? Revise any 
   comment that only says what you did without saying why.
   - Yes, the comments throughout the script explain both the action taken 
     and the business/analytical reasoning behind it.

3. After running describe(), what is the most important thing you noticed 
   about this dataset that you did not expect when you first loaded it?
   - The presence of negative values in quantity_sold or unit_price/revenue 
     highlighted that returns and refunds were mixed directly into sales data, 
     and extreme outlier quantities revealed placeholder entries 
     that would severely distort analytical models if not cleaned.
====================================================================
"""

import pandas as pd

# ==========================================
# EXERCISE 1: Inspect
# ==========================================
df = pd.read_csv("data/messy_sales.csv")

print("--- EXERCISE 1: INFO & NULLS ---")
print(df.info())
print("\nNull counts per column:")
print(df.isnull().sum())


# ==========================================
# EXERCISE 2: Standardize column names
# ==========================================
print("\n--- EXERCISE 2: STANDARDIZE COLUMNS ---")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Updated column list:", list(df.columns))


# ==========================================
# EXERCISE 3: Handle missing values
# ==========================================
print("\n--- EXERCISE 3: HANDLE MISSING VALUES ---")
print(f"Shape before dropping essential rows: {df.shape}")

df = df.dropna(subset=["order_id", "order_date"])

numeric_cols = ["unit_price", "quantity_sold"]
for col in numeric_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)

print(f"Shape after dropping essential rows: {df.shape}")
print("Null counts after handling missing values:")
print(df.isnull().sum())


# ==========================================
# EXERCISE 4: Fix types
# ==========================================
print("\n--- EXERCISE 4: FIX TYPES ---")

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
nat_count = df["order_date"].isna().sum()
print(f"Number of dates that became NaT: {nat_count}")

df["unit_price"] = (
    df["unit_price"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df["quantity_sold"] = df["quantity_sold"].astype(int)

df["revenue"] = (df["unit_price"] * df["quantity_sold"]).round(2)

print("\nUpdated Data Types:")
print(df.dtypes)


# ==========================================
# EXERCISE 5: Check duplicates
# ==========================================
print("\n--- EXERCISE 5: CHECK DUPLICATES ---")
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()

print(f"Shape after removing duplicates: {df.shape}")


# ==========================================
# EXERCISE 6: Describe & Clean Anomalies
# ==========================================
print("\n--- EXERCISE 6: DESCRIBE & ANOMALIES ---")
print(df.describe())

df = df[df["quantity_sold"] != 9999]
print(f"Shape after dropping placeholder quantities (9,999): {df.shape}")


# ==========================================
# EXERCISE 7: Save Cleaned Data
# ==========================================
print("\n--- EXERCISE 7: SAVE & RELOAD ---")

df.to_csv("data/clean_sales.csv", index=False)

df_reloaded = pd.read_csv("data/clean_sales.csv")
print("Reloaded without parse_dates (order_date is object):")
print(df_reloaded[["order_date"]].dtypes)

df_final = pd.read_csv("data/clean_sales.csv", parse_dates=["order_date"])
print("\nFinal Reloaded Info with parse_dates:")
print(df_final.info())
import os
import pandas as pd
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split

# 1. Setup
DATASET_NAME = "acs-income-ca"
STATE = "CA"
YEAR = "2018"
OUTPUT_DIR = f"datasets/{DATASET_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading {DATASET_NAME} (State: {STATE}, Year: {YEAR})...")

# 2. Download Data
data_source = ACSDataSource(survey_year=YEAR, horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=[STATE], download=True)

# 3. Extract Features, Label, and Group
features, label, group = ACSIncome.df_to_pandas(acs_data)

# 4. Combine into one DataFrame
# We want everything in one file so AIDE can choose what to use
df = pd.concat([features, label, group], axis=1)

# Remove duplicate columns if any (sometimes group is also in features)
df = df.loc[:, ~df.columns.duplicated()]

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 5. Split into Train/Test
# AIDE works best with explicit train/test files
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 6. Save
train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Saved train.csv to {train_path}")
print(f"Saved test.csv to {test_path}")

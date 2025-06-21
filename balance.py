import pandas as pd

# Load the Excel file
df = pd.read_excel("Balanced_Risk_Levels.xlsx")

# Check column names if unsure
print(df.columns.tolist())

# View class distribution
print(df['Risk_Level'].value_counts("No Risk"))


import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\ASUS\\Sem 7 B.E\\AIHC\\EDA_of_Healthcare_data\\EDA-HEALTHCARE - EDA-HEALTHCARE.csv')

# Display the first few rows 
print("First five rows of the dataset:")
print(df.head())

# Display basic information of the dataset
print("\nDataset Info:")
print(df.info())

# Display statistical summary of the dataset
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display value counts for categorical variables
print("\nUnivariate Non-Graphical Analysis:")
for column in df.select_dtypes(include=['object']).columns:
    print(f"\nValue counts for {column}:")
    print(df[column].value_counts())

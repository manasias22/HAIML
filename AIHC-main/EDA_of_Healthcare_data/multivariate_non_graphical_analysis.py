
import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\ASUS\\Sem 7 B.E\\AIHC\\EDA_of_Healthcare_data\\EDA-HEALTHCARE - EDA-HEALTHCARE.csv')

# Select only the numerical columns
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Compute and display the correlation matrix
print("\nMultivariate Non-Graphical Analysis:")
print("Correlation matrix for numerical variables:")
print(numerical_df.corr())

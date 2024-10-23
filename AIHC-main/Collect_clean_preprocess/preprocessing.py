import pandas as pd

# Step 1: Data Collection
# Load the diabetes dataset
df = pd.read_csv('diabetes - diabetes.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Step 2: Data Cleaning
# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Fill or drop missing values (this depends on your data and analysis needs)
# Example: Dropping rows with missing values
df.dropna(inplace=True)

# Check for duplicates
print("\nNumber of duplicate rows in the dataset:", df.duplicated().sum())

# Drop duplicate rows if any
df.drop_duplicates(inplace=True)

# Step 3: Data Integration
# Assuming you have another dataset related to diabetes (e.g., treatment data)
# Load additional dataset
# treatment_df = pd.read_csv('treatment_data.csv')

# Merge datasets on a common key (e.g., patient_id)
# df = pd.merge(df, treatment_df, on='patient_id', how='inner')

# Step 4: Data Transformation
# Example transformations: scaling, encoding categorical variables, etc.

# Convert categorical variables to numerical (if applicable)
# Example: One-hot encoding for a categorical variable 'Gender'
# df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Normalize or scale numerical features if necessary
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the transformed dataset
print("\nTransformed dataset:")
print(df.head())

# Save the cleaned and transformed dataset 
df.to_csv('cleaned_diabetes_data.csv', index=False)
print("\nCleaned and transformed dataset saved as 'cleaned_diabetes_data.csv'.")

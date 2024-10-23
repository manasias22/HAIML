
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:\\Users\\ASUS\\Sem 7 B.E\\AIHC\\EDA_of_Healthcare_data\\EDA-HEALTHCARE - EDA-HEALTHCARE.csv')

# Select only the numerical columns
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Pairplot for numerical variables
sns.pairplot(numerical_df, diag_kind='kde')
plt.suptitle('Pairplot for Numerical Variables', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Count plot for categorical variables
for column in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(8, 4))
    df[column] = df[column].astype(str)  # Ensure column is a string type
    top_categories = df[column].value_counts().nlargest(10).index  # Top 10 categories
    sns.countplot(y=column, data=df[df[column].isin(top_categories)])
    plt.title(f'Count plot for {column}')
    plt.show()
    
    # it may take some time to load the graphs in vs code :)

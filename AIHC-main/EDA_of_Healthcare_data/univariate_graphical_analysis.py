
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('C:\\Users\\ASUS\\Sem 7 B.E\\AIHC\\EDA_of_Healthcare_data\\EDA-HEALTHCARE - EDA-HEALTHCARE.csv')

# Plot histograms for numerical columns
df.hist(figsize=(10, 8), bins=30)
plt.suptitle('Histograms of Numerical Variables')
plt.show()

# Box plots for numerical columns to detect outliers
plt.figure(figsize=(12, 8))
df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(12, 8), sharex=False, sharey=False)
plt.suptitle('Boxplots of Numerical Variables')
plt.show()

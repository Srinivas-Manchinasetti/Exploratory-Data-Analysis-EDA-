# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")  # Update path if needed

# 1. Summary Statistics
print("Summary Statistics:\n")
print(df.describe(include='all'))  # Includes object (categorical) types
print("\nSurvival Distribution:\n")
print(df['Survived'].value_counts())

# 2. Histograms (only numeric columns)
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numeric_cols].hist(figsize=(10, 6), bins=20)
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# 3. Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplots of Numeric Features")
plt.show()

# 4. Pairplot (optional - subset to avoid overcrowding)
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.suptitle("Pairplot (Survived vs Features)", y=1.02)
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols + ['Survived']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 6. Inference Examples
print("\nInferences:")
print("- Age and Fare show variability; outliers are visible.")
print("- Survived correlates negatively with Pclass.")
print("- Parch and SibSp may indicate family travel and survival chances.")

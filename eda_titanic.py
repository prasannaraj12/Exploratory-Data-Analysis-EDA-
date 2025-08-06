# eda_titanic.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display first few rows
print("📌 First 5 rows of the dataset:")
print(df.head())

# Dataset info
print("\nℹ️ Dataset Info:")
df.info()

# Summary statistics
print("\n📊 Summary Statistics:")
print(df.describe(include='all'))

# Check missing values
print("\n🧼 Missing Values Count:")
print(df.isnull().sum())

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Handling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  # too many missing values

# Histograms for numeric features
df.hist(figsize=(12, 8), edgecolor='black')
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# Boxplots for detecting outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplots of Age and Fare")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for selected features
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'SibSp', 'Fare']], hue='Survived')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Skewness check
print("\n🌀 Skewness of numerical features:")
print(df.skew(numeric_only=True))

# Countplot: Survival
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

# Survival rate by Pclass
sns.barplot(data=df, x='Pclass', y='Survived')
plt.title("Survival Rate by Passenger Class")
plt.show()

# Age distribution by survival
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
plt.title("Age Distribution by Survival Status")
plt.show()

# Gender-based survival
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Survival Count by Gender")
plt.show()

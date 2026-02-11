import pandas as pd

# Load the processed data
df = pd.read_csv('data/processed/titanic_processed.csv')

# Select only numerical features for the model
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])  # Drop non-numeric columns

# Example feature: Family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Ensure that the target variable 'Survived' is included
df['Survived'] = df['Survived']  # Make sure we still have Survived included

# Save engineered features
df.to_csv('features/titanic_features.csv', index=False)

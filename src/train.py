import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load feature data
df = pd.read_csv('features/titanic_features.csv')

# Print the DataFrame to check its content
print("Columns in the dataset:")
print(df.columns)

# Ensure the target variable 'Survived' is defined
if 'Survived' not in df.columns:
    raise ValueError("Target variable 'Survived' is missing from the dataset.")

# Drop the target variable and ensure only numerical features are used
X = df.drop('Survived', axis=1)
y = df['Survived']

# Print the first few rows of X to identify any non-numeric data
print("Feature data:")
print(X.head())

# Optionally, check the dtypes of features
print("Data types:")
print(X.dtypes)

# Train the model on the filtered data
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'models/random_forest_model.pkl')

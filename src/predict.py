import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

# Load features for prediction
df = pd.read_csv('features/titanic_features.csv')

# Remove the target variable 'Survived' for prediction
if 'Survived' in df.columns:
    df = df.drop(columns=['Survived'])

# Generate predictions
predictions = model.predict(df)

# Save the predictions
prediction_df = pd.DataFrame(predictions, columns=['Survived'])
prediction_df.to_csv('results/predictions.csv', index=False)

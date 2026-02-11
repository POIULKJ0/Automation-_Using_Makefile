import pandas as pd
from sklearn.metrics import classification_report

# Load true values and predictions
df = pd.read_csv('features/titanic_features.csv')
true_values = df['Survived']
predictions = pd.read_csv('results/predictions.csv')

# Generate evaluation metrics
report = classification_report(true_values, predictions['Survived'])
with open('results/evaluation.txt', 'w') as f:
    f.write(report)

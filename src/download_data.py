import pandas as pd

# URL for the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Save the raw dataset
df.to_csv('data/raw/titanic.csv', index=False)


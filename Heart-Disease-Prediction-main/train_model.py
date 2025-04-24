import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('heart.csv')

# Select features and target
X = df[['age', 'cp', 'thalach']]
Y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')
print("Model trained and saved successfully!")

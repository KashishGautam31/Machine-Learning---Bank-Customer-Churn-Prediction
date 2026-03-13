# train_model.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns
df.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)

# Encode categorical columns
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# Separate features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("churn_model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully!")
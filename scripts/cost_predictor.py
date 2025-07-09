# cost_predictor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("prototype_data.csv")
X = df[["material_strength", "complexity", "expected_lifespan", "parts_availability_score"]]
y = df["cost"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R² score on test data: {score:.2f}")

# Save model
joblib.dump(model, "cost_predictor_model.pkl")
print("Cost predictor model saved as cost_predictor_model.pkl")

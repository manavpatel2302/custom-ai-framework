# data_preparation.py

import pandas as pd
import numpy as np

np.random.seed(42)

# Simulated dataset with 100 past prototypes:
data = {
    "design_id": np.arange(100),
    "material_strength": np.random.uniform(100, 300, 100),
    "complexity": np.random.uniform(1, 10, 100),
    "expected_lifespan": np.random.uniform(1, 5, 100),
    "parts_availability_score": np.random.uniform(0, 1, 100),
    "cost": np.random.uniform(2000, 5000, 100),  # Target for Cost Predictor
}

df = pd.DataFrame(data)
df.to_csv("prototype_data.csv", index=False)
print("Sample dataset saved as prototype_data.csv")

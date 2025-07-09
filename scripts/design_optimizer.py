# --- SNIP ---
import nevergrad as ng
import joblib
import numpy as np

# Load model
model = joblib.load("models/cost_predictor_model.pkl")

# Define objective
def objective(x):
    material_strength, complexity, expected_lifespan, parts_availability = x
    input_features = np.array([[material_strength, complexity, expected_lifespan, parts_availability]])
    predicted_cost = model.predict(input_features)[0]
    return predicted_cost

# Bounds
lower = np.array([100, 1, 1, 0])
upper = np.array([300, 10, 5, 1])
midpoint = (lower + upper) / 2

# Parametrization
parametrization = ng.p.Array(shape=(4,))
parametrization.value = midpoint  # ✅ initial value inside bounds
parametrization.set_bounds(lower=lower, upper=upper)

# Optimizer
optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=100)
recommendation = optimizer.minimize(objective)

best_params = recommendation.value
best_cost = objective(best_params)

print("\nOptimal Design Parameters Found:")
print(f"Material Strength: {best_params[0]:.2f}")
print(f"Complexity: {best_params[1]:.2f}")
print(f"Expected Lifespan: {best_params[2]:.2f}")
print(f"Parts Availability Score: {best_params[3]:.2f}")
print(f"Predicted Cost: ₹{best_cost:.2f}")

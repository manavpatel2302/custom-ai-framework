import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scripts.parts_recommender import recommend_parts

# Load trained model
model = joblib.load("models/cost_predictor_model.pkl")

st.set_page_config(page_title="AI R&D Framework", layout="centered")
st.title("📊 Cost Estimator & Parts Recommender")

st.write("Upload a CSV file with the following columns:")
st.code("material_strength, complexity, expected_lifespan, parts_availability_score")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ["material_strength", "complexity", "expected_lifespan", "parts_availability_score"]
        
        if not all(col in df.columns for col in required_cols):
            st.error("❌ Your file must contain the required columns.")
        else:
            # Predict cost
            X = df[required_cols]
            df["predicted_cost"] = model.predict(X)
            
            # Recommend parts
            df["parts_recommendation"] = df["parts_availability_score"].apply(recommend_parts)
            
            st.success("✅ Predictions complete!")
            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
else:
    st.info("👆 Upload a file above to begin.")

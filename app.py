import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import os
import io

# ────────────────────────────────────────────────
# Title & Description
# ────────────────────────────────────────────────
st.set_page_config(page_title="Habitat Success Simulator", layout="wide")

st.title("Habitat Success Simulator 🐺🌳")
st.markdown("""
This AI-powered tool predicts **animal survival rates** based on species, landcover type, 
and reproductive success — using public-data-inspired patterns (e.g., USGS NLCD landcover classes, 
ecology trends from GBIF/USFWS).

**How it works**  
• Enter values → AI model predicts survival (0–1 scale)  
• Lower survival often occurs in degraded/urban habitats  
• Expand later with real CSV uploads from data.gov, GBIF, etc.
""")

# ────────────────────────────────────────────────
# Sample Data (realistic based on ecology studies)
# ────────────────────────────────────────────────
data = {
    'Species': ['Wolf', 'Wolf', 'Wolf', 'Caribou', 'Caribou', 'Caribou', 'Tortoise', 'Tortoise', 'Tortoise', 'Bear', 'Bear'],
    'Landcover': ['Forest', 'Grassland', 'Urban', 'Forest', 'Grassland', 'Urban', 'Desert', 'Forest', 'Urban', 'Forest', 'Urban'],
    'Reproductive_Success': [0.80, 0.70, 0.50, 0.75, 0.65, 0.40, 0.60, 0.50, 0.30, 0.85, 0.55],
    'Survival_Rate': [0.70, 0.60, 0.40, 0.65, 0.55, 0.30, 0.55, 0.45, 0.25, 0.75, 0.50]
}
df = pd.DataFrame(data)

# ────────────────────────────────────────────────
# Prepare encoders & data for model
# ────────────────────────────────────────────────
@st.cache_data
def prepare_data():
    le_species = LabelEncoder()
    le_landcover = LabelEncoder()
    df['Species_enc'] = le_species.fit_transform(df['Species'])
    df['Landcover_enc'] = le_landcover.fit_transform(df['Landcover'])
    return le_species, le_landcover, df

le_species, le_landcover, df = prepare_data()

X = torch.tensor(df[['Species_enc', 'Landcover_enc', 'Reproductive_Success']].values, dtype=torch.float32)
y = torch.tensor(df['Survival_Rate'].values, dtype=torch.float32).unsqueeze(1)

# ────────────────────────────────────────────────
# Neural Network Model
# ────────────────────────────────────────────────
class SurvivalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ────────────────────────────────────────────────
# Load or Train Model (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def get_model():
    model = SurvivalPredictor()
    model_path = "survival_model.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        with st.spinner("Training simple AI model on sample data (one-time)..."):
            for epoch in range(300):  # more epochs → slightly better fit
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), model_path)
    return model

model = get_model()

# ────────────────────────────────────────────────
# User Interface
# ────────────────────────────────────────────────
st.subheader("Make a Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    species = st.selectbox("Species", options=sorted(df['Species'].unique()), index=0)

with col2:
    landcover = st.selectbox("Landcover Type", options=sorted(df['Landcover'].unique()), index=0)

with col3:
    rep_success = st.slider("Reproductive Success (0.0 = very low → 1.0 = excellent)", 
                            min_value=0.0, max_value=1.0, value=0.70, step=0.05)

if st.button("Predict Survival Rate", type="primary"):
    try:
        species_enc = le_species.transform([species])[0]
        landcover_enc = le_landcover.transform([landcover])[0]

        input_tensor = torch.tensor([[species_enc, landcover_enc, rep_success]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.success(f"**Predicted Survival Rate: {prediction:.2%}**")

        st.markdown(f"""
        **Interpretation**  
        - {prediction:.0%} survival suggests **{ 'good' if prediction > 0.6 else 'moderate' if prediction > 0.4 else 'poor' }** habitat fit.  
        - Urban/developed landcover often reduces survival (seen in real ecology data).  
        - Try different scenarios to simulate habitat change impacts!
        """)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ────────────────────────────────────────────────
# Data Table & Upload for Future Expansion
# ────────────────────────────────────────────────
with st.expander("View Sample Training Data & Upload Your Own CSV (advanced)"):
    st.dataframe(df.style.format({"Reproductive_Success": "{:.2f}", "Survival_Rate": "{:.2f}"}))
    
    uploaded_file = st.file_uploader("Upload real data CSV (columns: Species, Landcover, Reproductive_Success, Survival_Rate)", type="csv")
    if uploaded_file:
        st.info("CSV uploaded! In a full version, we would retrain the model here.")
        # Future: df = pd.read_csv(uploaded_file); retrain logic

# ────────────────────────────────────────────────
# Footer / Next Steps
# ────────────────────────────────────────────────
st.markdown("---")
st.caption("Built for beginners • Powered by Streamlit + PyTorch • Expand with real public datasets (USGS NLCD, GBIF, USFWS)")

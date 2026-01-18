# SiteSync-AI-

## Smart Construction Management System
A machine learning-powered application for construction resource planning, labour allocation, and risk management.

### Features
- **Material Prediction (Panel 1):** Predicts weekly needs for Cement, Steel, Bricks, and Sand based on site stage and inputs.
- **Labour Prediction (Panel 2):** Calculates required Skilled and Unskilled workforce based on task and material volume.
- **Risk Assessment (Panel 3):** Tri-Model engine (Shortfall -> Score -> Class) to predict project delay risk.
- **Optimization Engine (Panel 4):** Uses Linear Programming (PuLP) to generate procurement plans with safety buffers.

### Setup
1. `pip install -r requirements.txt`
2. `python -m streamlit run frontend/app.py`

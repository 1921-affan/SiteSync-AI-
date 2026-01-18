# 🏗️ SiteSync AI - Smart Construction Management System

**SiteSync AI** is an advanced Machine Learning and Optimization platform designed to streamline construction resource planning. It integrates **Predictive AI** with **Linear Programming** to forecast material needs, allocate labour efficiently, assess project risks, and generate cost-optimized procurement plans.

---

## 🚀 Key Features

### 1. Integrated 4-Panel Prediction System
The application is built around a logical flow of **4 Intelligent Panels**:

*   **Panel 1: Material Prediction (Site Context)**
    *   **Input:** Site Area, Floor Count, Construction Stage (1-5), Property Type.
    *   **Output:** Precise quantity predictions for **Cement, Steel, Bricks, and Sand**.
    *   **Model:** Random Forest Regressor (Trained on 5,000+ construction weeks).

*   **Panel 2: Labour Allocation (Task Aware)**
    *   **Input:** Material Volume (from Panel 1), Primary Task (e.g., Concrete Pour vs. Excavation), Productivity Rate.
    *   **Output:** Required **Skilled Masons** and **Unskilled Helpers**.
    *   **Intelligence:** Context-aware; "Excavation" requests helpers, "Brickwork" requests masons. Adjusts for productivity norms.

*   **Panel 3: Risk Assessment (Tri-Model Engine)**
    *   **Input:** Material/Labour Availability, Weather, Past Delays, Progress Gap.
    *   **Output:** 
        *   **Shortfall Est:** Expected worker shortage.
        *   **Risk Score:** 0-15 scale.
        *   **Risk Class:** **Low / Medium / High** (Determines safety buffers).
    *   **Model:** Chained Multi-Output Regressors + Classifier.

*   **Panel 4: Resource Optimization (PuLP Engine)**
    *   **Input:** Predictions from Panels 1-3 + **Current Inventory Levels**.
    *   **Output:** Cost-minimized Procurement Plan.
    *   **Logic:** 
        *   Subtracts existing inventory from demand.
        *   Applies safety buffers based on Risk Class (Low=1.0x, High=1.2x).
        *   Decides between **Regular vs. Express** delivery to meet deadlines.

---

## 🛠️ Technology Stack
*   **Frontend:** Streamlit (Python)
*   **Machine Learning:** Scikit-Learn, LightGBM, RandomForest
*   **Optimization:** PuLP (Linear Programming)
*   **Data Processing:** Pandas, NumPy
*   **Backend:** Modular Python Scripts (`predict_*.py`)

---

## 📦 Installation & Setup

### Prerequisites
*   Python 3.8+
*   pip

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/1921-affan/SiteSync-AI-.git
    cd SiteSync-AI-
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python -m streamlit run frontend/app.py
    ```

---

## 📖 Usage Guide

### Dashboard View
1.  **Context:** Enter the Current Week, Project ID, and Site details.
2.  **Stage:** Select strictly from the 5 defined stages (Foundation -> Finishing).
3.  **Run Prediction:** Click **"RUN INTEGRATED AI PREDICTION"**.
4.  **Review:** See predicted Materials, Labour counts, and Risk Status.

### Optimization Engine (Page 2)
1.  **Navigate:** Select "Optimization Engine" from the Sidebar.
2.  **Inventory:** Enter your current stock of Cement, Steel, etc.
3.  **Generate Plan:** Click **"GENERATE PROCUREMENT PLAN"**.
4.  **Result:** View the "Net Order Qty" and total estimated weekly cost.

---

## 📂 Project Structure
```
├── backend/
│   ├── data/               # Raw and Static datasets (Costs, Train Data)
│   ├── models/             # Trained .pkl models and encoders
│   ├── scripts/            # Core Logic
│   │   ├── predict_materials.py  # Panel 1 Inference
│   │   ├── predict_labour.py     # Panel 2 Inference
│   │   ├── predict_risk.py       # Panel 3 Inference
│   │   └── optimize_resources.py # Panel 4 PuLP Engine
│   └── Training/           # Model training scripts
├── frontend/
│   └── app.py              # Main Streamlit Application
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💡 System Logic (Example)
*   If **Risk is High**, the Optimization Engine automatically adds a **20% Safety Buffer** to material orders.
*   If **Inventory > Demand**, the Net Order Quantity is **0**, saving costs.
*   If **Task = Excavation**, the Labour Model ignores Masonry requirements, aligning with reality.

---
**SiteSync AI** — *Predict. Plan. Optimize.*

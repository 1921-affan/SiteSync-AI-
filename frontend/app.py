import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# ---------------------------------------------------------
# 1. SETUP & IMPORTS
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, '..')
sys.path.append(backend_path)

try:
    from backend.scripts import predict_materials
    from backend.scripts import predict_labour
    from backend.scripts import predict_risk
    from backend.scripts import optimize_resources # <--- NEW IMPORT
    BACKEND_ONLINE = True
except ImportError as e:
    st.error(f"❌ Backend Scripts Not Found: {e}")
    BACKEND_ONLINE = False

st.set_page_config(page_title="Construction AI Manager Pro", layout="wide", page_icon="🏗️")

# Custom CSS
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    .panel-header { font-size: 20px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
    .stButton>button { width: 100%; background-color: #2ecc71; color: white; font-weight: bold; font-size: 18px; padding: 10px; }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-med { color: #f39c12; font-weight: bold; }
    .risk-low { color: #27ae60; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. NAVIGATION & STATE
# ---------------------------------------------------------
st.sidebar.title("🏗️ Site Manager")
page = st.sidebar.radio("Navigate", ["📊 Prediction Dashboard", "⚙️ Optimization Engine"])
st.sidebar.markdown("---")

# Global Context (Restored)
proj_id = st.sidebar.text_input("Project ID", "PROJ-2025-001")
week_num = st.sidebar.number_input("Current Week", min_value=1, value=15)
st.sidebar.markdown("---")

if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = None

if not BACKEND_ONLINE:
    st.stop()

# =========================================================
# PAGE 1: PREDICTION DASHBOARD
# =========================================================
if page == "📊 Prediction Dashboard":
    st.title("🚀 Construction AI Dashboard")
    st.markdown("### Integrated 4-Panel Prediction System")
    
    # --- INPUTS ---
    with st.container():
        st.markdown('<div class="panel-header">1️⃣ Panel 1: Material (Site Context)</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        site_area = c1.number_input("Site Area (sqft)", 1000, 50000, 2400)
        dims = c1.selectbox("Dimensions", ["30x40", "40x60", "50x80", "100x100"])
        floors = c2.number_input("Total Floors", 1, 50, 4)
        floors_comp = c2.number_input("Floors Completed", 0, 50, 2)
        stage_options = [
            "Stage 1: Foundation (Footings, Rebar below ground)",
            "Stage 2: Plinth & Backfilling (Beams, Compaction)",
            "Stage 3: Structural Frame (Columns, Beams, Slabs)",
            "Stage 4: Masonry & Roof (Walls, Lintels)",
            "Stage 5: Building Envelope (Plastering, Finishing)"
        ]
        stage = c3.selectbox("Stage", stage_options, index=2)
        prop_type = c3.selectbox("Type", ["1BHK", "2BHK", "3BHK", "Godown"], index=1)
        weather = c4.selectbox("Weather", ["Normal", "Light_Rain", "Heavy_Rain", "High_Wind"])
        lost_days = c4.number_input("Lost Days", 0, 30, 0)
        pace = c4.radio("Pace", ["Normal", "Fast_Track"], horizontal=True)

    with st.container():
        st.markdown("---")
        st.markdown('<div class="panel-header">2️⃣ Panel 2: Labour (Task)</div>', unsafe_allow_html=True)
        l1, l2 = st.columns(2)
        task_type = l1.selectbox("Primary Task", ["Concrete_Pour", "Brick_Masonry", "Plastering", "Rebar_Binding"])
        prod_rate = l2.slider("Productivity Rate", 0.5, 1.5, 1.0)

    with st.container():
        st.markdown("---")
        st.markdown('<div class="panel-header">3️⃣ Panel 3: Risk Factors</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        mat_status = r1.selectbox("Material Availability", ["Available", "Limited", "Shortage"])
        lab_status = r2.selectbox("Labour Availability", ["Sufficient", "Moderate", "Critical"])
        hist_delays = r3.number_input("Past Delays", 0, 10, 2)
        prog_gap = r4.number_input("Progress Gap (%)", -50.0, 50.0, 12.0)

    # --- EXECUTION ---
    st.markdown("---")
    if st.button("🧠 RUN INTEGRATED AI PREDICTION"):
        try:
            # Step 1: Material
            stage_map = {
                "Stage 1: Foundation (Footings, Rebar below ground)": 1,
                "Stage 2: Plinth & Backfilling (Beams, Compaction)": 2,
                "Stage 3: Structural Frame (Columns, Beams, Slabs)": 3,
                "Stage 4: Masonry & Roof (Walls, Lintels)": 4,
                "Stage 5: Building Envelope (Plastering, Finishing)": 5
            }
            mat_inputs = {
                "week_number": 15, "site_area_sqft": site_area, "total_floors": floors,
                "floors_completed": floors_comp, "construction_stage": stage_map[stage],
                "property_type": prop_type, "lost_days": lost_days,
                "weather_condition": weather, "site_dimensions": dims, "work_pace": pace
            }
            res_mat = predict_materials.predict_material_needs(mat_inputs)
            
            # Step 2: Labour
            lab_inputs = {
                "week_number": 15, "site_area_sqft": site_area, "construction_stage": stage_map[stage],
                "floors_completed": floors_comp, "is_complex_design": 0, "productivity_rate": prod_rate,
                "req_cement_bags": res_mat['req_cement_bags'], "req_bricks_nos": res_mat['req_bricks_nos'],
                "req_steel_kg": res_mat['req_steel_kg'], "req_sand_tons": res_mat['req_sand_tons'],
                "task_type": task_type
            }
            res_lab = predict_labour.predict_labour_needs(lab_inputs)
            
            # Step 3: Risk
            risk_inputs = {
                "req_skilled_labour": res_lab['req_skilled_labour'],
                "req_unskilled_labour": res_lab['req_unskilled_labour'],
                "material_avail_status": mat_status, "labour_avail_status": lab_status,
                "weather_condition": weather, "progress_gap_pct": prog_gap,
                "hist_material_delay_count": hist_delays, "construction_stage": stage_map[stage]
            }
            res_risk = predict_risk.predict_risk_metrics(risk_inputs)
            
            # SAVE STATE
            full_results = {**res_mat, **res_lab, **res_risk}
            st.session_state.last_predictions = full_results
            st.success("✅ Prediction Complete! Results Saved for Optimization.")

            # DISPLAY
            c1, c2, c3 = st.columns(3)
            c1.subheader("📦 Materials"); c1.write(res_mat)
            c2.subheader("👷 Labour"); c2.write(res_lab)
            
            r_color = {"Low": "green", "Med": "orange", "High": "red"}.get(res_risk['risk_class'], "black")
            c3.subheader("🚨 Risk Metrics")
            c3.metric("Shortfall Est", f"{res_risk['labour_shortfall_est']} Workers")
            c3.metric("Risk Score", f"{res_risk['risk_score']}/15")
            c3.markdown(f"**Class:** :{r_color}[{res_risk['risk_class']}]")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# PAGE 2: OPTIMIZATION ENGINE
# =========================================================
elif page == "⚙️ Optimization Engine":
    st.title("⚙️ Resource Optimization Engine (PuLP)")
    st.markdown("Uses **Linear Programming** to minimize cost while meeting safety buffers defined by the Risk Class.")
    
    if st.session_state.last_predictions is None:
        st.warning("⚠️ No prediction data found. Please run the 'Prediction Dashboard' first.")
    else:
        preds = st.session_state.last_predictions
        # Map Risk to String for Display
        risk_raw = preds.get('risk_class', 'Unknown')
        risk_map = {0: 'Low', 1: 'Medium', 2: 'High', 'Low': 'Low', 'Medium': 'Medium', 'High': 'High'}
        risk_label = risk_map.get(risk_raw, str(risk_raw))
        
        st.info(f"Using Prediction Data (Risk: {risk_label})")
        
        # Inventory Input Section
        st.markdown("### 📦 Current Inventory (Subtracts from Demand)")
        i1, i2, i3, i4 = st.columns(4)
        inv_cement = i1.number_input("Stock: Cement (Bags)", 0, 5000, 0)
        inv_steel = i2.number_input("Stock: Steel (Kg)", 0, 10000, 0)
        inv_bricks = i3.number_input("Stock: Bricks (Nos)", 0, 50000, 0)
        inv_sand = i4.number_input("Stock: Sand (Tons)", 0, 500, 0)
        
        preds['inventory'] = {
            "Cement": inv_cement,
            "Steel": inv_steel,
            "Bricks": inv_bricks,
            "Sand": inv_sand
        }

        if st.button("🚀 GENERATE PROCUREMENT PLAN"):
            try:
                df_plan = optimize_resources.optimize_procurement(preds)
                
                # Metrics
                total_cost = df_plan['Est_Cost'].sum()
                st.metric("💰 Total Estimated Weekly Cost", f"₹ {total_cost:,.2f}")
                
                # Display Plan
                st.subheader("📋 Optimized Procurement Plan")
                
                # Split Material vs Labour tables
                st.markdown("#### 📦 Material Orders")
                # Filter for materials and show logical columns
                df_mat = df_plan[df_plan['Category']=='Material']
                st.dataframe(
                    df_mat[['Resource', 'Total_Need_With_Buffer', 'Inventory_Used', 'Net_Order_Qty', 'Sourced_Regular', 'Sourced_Premium', 'Est_Cost', 'Note']], 
                    use_container_width=True
                )
                
                st.markdown("#### 👷 Workforce Hiring")
                st.dataframe(df_plan[df_plan['Category']=='Labour'], use_container_width=True)
                
                # Highlight Alerts
                alerts = df_plan[df_plan['Note'].str.contains("⚠️")]
                if not alerts.empty:
                    st.error("⚠️ CRITICAL ALERTS: Premium sourcing required!")
                    st.table(alerts[['Resource', 'Note', 'Est_Cost']])
                    
            except Exception as e:
                st.error(f"Optimization Error: {e}")
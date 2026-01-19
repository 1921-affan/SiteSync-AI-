import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from fpdf import FPDF
import base64

# --- PATH SETUP ---
# Ensure we can find the backend scripts
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, '..', 'backend', 'scripts')
sys.path.append(backend_path)

# --- BACKEND IMPORTS ---
import predict_materials
import predict_labour
import predict_risk
import optimize_resources

def generate_pdf_report(df_plan, total_cost, risk_class, proj_id, week, preds):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SiteSync AI - Weekly Project Report", ln=True, align='C')
    pdf.ln(5)
    
    # Context
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Project ID: {proj_id}", ln=True)
    pdf.cell(200, 8, txt=f"Week Number: {week}", ln=True)
    pdf.cell(200, 8, txt=f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)
    
    # --- PANEL 1 & 2: PREDICTIONS ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="1. AI Requirement Predictions", ln=True)
    pdf.set_font("Arial", size=11)
    
    # Material
    pdf.set_font("Arial", 'B', 11); pdf.cell(200, 8, txt="Material Needs (Site Context):", ln=True); pdf.set_font("Arial", size=11)
    pdf.cell(100, 6, txt=f" - Cement: {preds.get('req_cement_bags', 0)} Bags", ln=True)
    pdf.cell(100, 6, txt=f" - Steel: {preds.get('req_steel_kg', 0)} Kg", ln=True)
    pdf.cell(100, 6, txt=f" - Bricks: {preds.get('req_bricks_nos', 0)} Nos", ln=True)
    pdf.cell(100, 6, txt=f" - Sand: {preds.get('req_sand_tons', 0)} Tons", ln=True)
    pdf.ln(2)
    
    # Labour
    pdf.set_font("Arial", 'B', 11); pdf.cell(200, 8, txt="Workforce Needs (Task Based):", ln=True); pdf.set_font("Arial", size=11)
    pdf.cell(100, 6, txt=f" - Skilled Masons: {preds.get('req_skilled_labour', 0)}", ln=True)
    pdf.cell(100, 6, txt=f" - Unskilled Helpers: {preds.get('req_unskilled_labour', 0)}", ln=True)
    pdf.ln(5)

    # --- PANEL 3: RISK ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="2. Risk Assessment", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(100, 6, txt=f"Risk Class: {preds.get('risk_class', 'Unknown')}", ln=True)
    pdf.cell(100, 6, txt=f"Risk Score: {preds.get('risk_score', 0)}/15", ln=True)
    pdf.cell(100, 6, txt=f"Est. Labour Shortfall: {preds.get('labour_shortfall_est', 0)} Workers", ln=True)
    pdf.ln(5)

    # --- INVENTORY ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="3. Current Inventory Levels", ln=True)
    pdf.set_font("Arial", size=11)
    inv = preds.get('inventory', {})
    pdf.cell(100, 6, txt=f" - Cement: {inv.get('Cement', 0)} Bags", ln=True)
    pdf.cell(100, 6, txt=f" - Steel: {inv.get('Steel', 0)} Kg", ln=True)
    pdf.cell(100, 6, txt=f" - Bricks: {inv.get('Bricks', 0)} Nos", ln=True)
    pdf.cell(100, 6, txt=f" - Sand: {inv.get('Sand', 0)} Tons", ln=True)
    pdf.ln(5)
    
    # --- PANEL 4: PROCUREMENT ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="4. Optimized Procurement Plan", ln=True)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(35, 8, "Category", 1)
    pdf.cell(45, 8, "Resource", 1)
    pdf.cell(25, 8, "Net Qty", 1)
    pdf.cell(35, 8, "Est Cost (INR)", 1)
    pdf.cell(40, 8, "Note", 1)
    pdf.ln()
    
    # Table Content
    pdf.set_font("Arial", size=10)
    for index, row in df_plan.iterrows():
        pdf.cell(35, 8, str(row['Category']), 1)
        pdf.cell(45, 8, str(row['Resource']), 1)
        pdf.cell(25, 8, str(row['Net_Order_Qty']), 1)
        pdf.cell(35, 8, str(row['Est_Cost']), 1)
        pdf.cell(40, 8, str(row['Note']), 1)
        pdf.ln()
        
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Total Weekly Cost: INR {total_cost:,.2f}", ln=True)
    
    # Save Logic (Return binary)
    return pdf.output(dest='S').encode('latin-1')

# ... (Existing Code)

# ---------------------------------------------------------
# 2. NAVIGATION & STATE
# ---------------------------------------------------------
st.sidebar.title("🏗️ Site Manager")
page = st.sidebar.radio("Navigate", ["📊 Prediction Dashboard", "⚙️ Optimization Engine", "📈 Analytics Dashboard"])
st.sidebar.markdown("---")

# Global Context (Restored)
proj_id = st.sidebar.text_input("Project ID", "PROJ-2025-001")
week_num = st.sidebar.number_input("Current Week", min_value=1, value=15)
st.sidebar.markdown("---")

# =========================================================
# PAGE 1: PREDICTION DASHBOARD
# =========================================================
if page == "📊 Prediction Dashboard":
    st.title("🤖 Construction AI Manager")

    # --- PANEL 1: SITE CONTEXT ---
    # --- PANEL 1: SITE CONTEXT ---
    with st.expander("1. Site & Material Context", expanded=True):
        c1, c2 = st.columns(2)
        area = c1.number_input("Plot/Footprint Area (sqft)", min_value=500, value=1200, help="Enter ground area only (e.g., 600 for 20x30). Model multiplies this by floors internally.")
        floors = c2.number_input("Floors to Build", min_value=1, value=2)
        stage_options = [
            "Stage 1: Foundation Construction",
            "Stage 2: Plinth Beam & Backfilling",
            "Stage 3: Structural Frame & Slabs",
            "Stage 4: Masonry & Roof Structure",
            "Stage 5: Building Envelope"
        ]
        stage = st.selectbox("Construction Stage", options=stage_options, index=2)
        
        # Stage Descriptions
        stage_desc = {
            "Stage 1: Foundation Construction": "Footings, Concrete, Rebar below ground (High Cement/Steel).",
            "Stage 2: Plinth Beam & Backfilling": "Beams, Soil compaction, rising from ground level.",
            "Stage 3: Structural Frame & Slabs": "Columns, Beams, All Floor Slabs (Max Steel/Concrete, No Bricks).",
            "Stage 4: Masonry & Roof Structure": "Internal/External Walls, Parapets, Lintels (Max Bricks, Low Steel).",
            "Stage 5: Building Envelope": "Plastering, Flooring, Waterproofing (Finishing work)."
        }
        st.info(f"ℹ️ {stage_desc[stage]}")
        
        c3, c4 = st.columns(2)
        prop_type = c3.selectbox("Property Type", ["1BHK", "2BHK", "3BHK", "4BHK", "Villa", "Godown", "Shop", "Other"])
        dims = c4.selectbox("Site Dimensions (ft)", ["20x30", "30x40", "30x50", "40x60", "50x80", "Other"])
        
        c5, c6 = st.columns(2)
        lost_days = c5.number_input("Lost Days (Weather/Delays)", min_value=0, value=0)
        work_pace = c6.selectbox("Work Pace", ["Select...", "Normal", "Fast_Track"], index=0)
        
        c7, c8 = st.columns(2)
        floors_done = c7.number_input("Floors Completed So Far", min_value=0, value=0, help="Actual progress independent of stage.")
        # Derived: Complexity
        is_complex = True if prop_type in ["Villa", "Commercial", "Other"] else False

    # --- PANEL 2: TASK & LABOUR CONTEXT ---
    with st.expander("2. Workforce & Task Context", expanded=True):
        task_type = st.selectbox("Primary Task This Week", [
            "Excavation", "Foundation_Pour", "Brickwork_Walls", "Concrete_Slab", "Plastering", "Tile_Work"
        ], index=2)
        productivity = st.slider("Team Productivity Rate", 0.5, 1.5, 1.0, 0.1)

    # --- PANEL 3: RISK FACTORS ---
    with st.expander("3. Risk Diagnostics", expanded=True):
        r1, r2, r3 = st.columns(3)
        mat_avail = r1.selectbox("Material Availability", ["Available", "Limited", "Shortage"])
        lab_avail = r2.selectbox("Labour Availability", ["Sufficient", "Moderate", "Critical"])
        weather = r3.selectbox("Weather Forecast", ["Normal", "Light_Rain", "Heavy_Rain", "High_Wind", "Heatwave"])
        
        hist_delays = st.number_input("Past Material Delays (Count)", 0, 10, 0)
        prog_gap = st.number_input("Schedule Gap (%)", -50, 50, 5, help="Positive = Behind Schedule")

    # --- RUN BUTTON ---
    stage_map = {
        "Stage 1: Foundation Construction": 1,
        "Stage 2: Plinth Beam & Backfilling": 2,
        "Stage 3: Structural Frame & Slabs": 3,
        "Stage 4: Masonry & Roof Structure": 4,
        "Stage 5: Building Envelope": 5
    }

    if st.button("🚀 RUN INTEGRATED AI PREDICTION"):
        if work_pace == "Select...":
            st.error("⚠️ Please select a valid Work Pace. All fields are required for accurate prediction.")
            st.stop()
            
        with st.spinner("Analyzing Site Data..."):
            # 1. Predict Materials
            # Function: predict_material_needs(inputs)
            # Inputs: week_number, site_area_sqft, total_floors, floors_completed, construction_stage, property_type, lost_days, weather_condition, site_dimensions, work_pace
            mat_inputs = {
                "week_number": week_num,
                "site_area_sqft": area,
                "total_floors": floors,
                "floors_completed": floors_done, # Updated from UI (Manual Input)
                "construction_stage": stage_map.get(stage, 1),
                "property_type": prop_type,
                "lost_days": lost_days,
                "weather_condition": weather,
                "site_dimensions": dims,
                "work_pace": work_pace
            }
            mat_preds = predict_materials.predict_material_needs(mat_inputs)
            
            # 2. Predict Labour (Chained)
            # Function: predict_labour_needs(inputs)
            lab_inputs = {
                "week_number": week_num,
                "site_area_sqft": area,
                "construction_stage": stage_map.get(stage, 1),
                "floors_completed": max(0, stage_map.get(stage, 0) - 1),
                "is_complex_design": 1 if is_complex else 0,
                "productivity_rate": productivity,
                "req_cement_bags": mat_preds['req_cement_bags'],
                "req_bricks_nos": mat_preds['req_bricks_nos'],
                "req_steel_kg": mat_preds['req_steel_kg'],
                "req_sand_tons": mat_preds['req_sand_tons'],
                "task_type": task_type
            }
            lab_preds = predict_labour.predict_labour_needs(lab_inputs)
            
            # 3. Predict Risk (Tri-Model)
            # Function: predict_risk_metrics(inputs)
            risk_inputs = {
                "req_skilled_labour": lab_preds['req_skilled_labour'],
                "req_unskilled_labour": lab_preds['req_unskilled_labour'],
                "material_avail_status": mat_avail,
                "labour_avail_status": lab_avail,
                "weather_condition": weather,
                "progress_gap_pct": prog_gap,
                "hist_material_delay_count": hist_delays,
                "construction_stage": stage_map.get(stage, 1)
            }
            risk_preds = predict_risk.predict_risk_metrics(risk_inputs)
            
            # --- MANUAL ADJUSTMENT BUFFER ---
            # User request: Add safety buffer (+1 Skilled, +2 Unskilled)
            lab_preds['req_skilled_labour'] += 1
            lab_preds['req_unskilled_labour'] += 2
            
            # COMBINE & STORE RESULTS
            full_results = {**mat_preds, **lab_preds, **risk_preds}
            
            # Save to Session State for Optimization Page
            st.session_state['latest_preds'] = full_results
            st.session_state['proj_id'] = proj_id
            st.session_state['week_num'] = week_num
            
            # --- LOGGING ---
            from datetime import datetime
            
            log_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Project_ID": proj_id,
                "Week_Number": week_num,
                "Stage": stage,
                "Progress_Gap_Pct": prog_gap,
                **full_results 
            }
            
            # Save to CSV
            project_root = os.path.dirname(backend_path)
            log_dir = os.path.join(project_root, 'outputs', 'pipeline_results')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'prediction_history.csv')
            
            df_log = pd.DataFrame([log_data])
            if not os.path.exists(log_file):
                df_log.to_csv(log_file, index=False)
            else:
                df_log.to_csv(log_file, mode='a', header=False, index=False)
            
            st.success(f"✅ Prediction Complete! Results Logged.")
            
            # --- DISPLAY DASHBOARD ---
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.info("🧱 Material Needs")
                st.write(f"**Cement:** {mat_preds['req_cement_bags']} Bags")
                st.write(f"**Steel:** {mat_preds['req_steel_kg']} Kg")
                st.write(f"**Bricks:** {mat_preds['req_bricks_nos']} Nos")
                st.write(f"**Sand:** {mat_preds['req_sand_tons']} Tons")
            
            with c2:
                st.info("👷 Workforce Plan")
                st.write(f"**Skilled Masons:** {lab_preds['req_skilled_labour']}")
                st.write(f"**Helpers:** {lab_preds['req_unskilled_labour']}")
                st.caption(f"Task: {task_type}")
            
            with c3:
                risk_cls = risk_preds['risk_class']
                color = "green" if risk_cls=="Low" else "orange" if risk_cls=="Medium" else "red"
                if risk_cls == "High":
                    st.error(f"🚨 Risk Status: {risk_cls}")
                elif risk_cls == "Medium":
                    st.warning(f"Risk: {risk_cls}")
                else:
                    st.success(f"Risk: {risk_cls}")
                
                st.metric("Risk Score", f"{risk_preds['risk_score']}/15")
                st.write(f"**Est Shortfall:** {risk_preds['labour_shortfall_est']} Workers")

# =========================================================
# PAGE 2: OPTIMIZATION ENGINE
# =========================================================
elif page == "⚙️ Optimization Engine":
    st.title("⚙️ Resource Optimization Engine")

    # Check context
    if 'latest_preds' not in st.session_state:
        st.warning("⚠️ Please run a prediction on the Dashboard first!")
    else:
        preds = st.session_state['latest_preds']
        proj_id = st.session_state.get('proj_id', 'Unknown')
        week_num = st.session_state.get('week_num', 0)
        
        st.success(f"Context Loaded: {proj_id} | Week {week_num}")
        
        # --- INVENTORY INPUTS ---
        st.subheader("📦 Current Site Inventory")
        c1, c2, c3, c4 = st.columns(4)
        inv_cement = c1.number_input("Cement (Bags)", 0, 1000, 0)
        inv_steel = c2.number_input("Steel (Kg)", 0, 5000, 0)
        inv_bricks = c3.number_input("Bricks (Nos)", 0, 10000, 0)
        inv_sand = c4.number_input("Sand (Tons)", 0, 100, 0)
        
        # Update Predictions with Inventory
        preds['inventory'] = {
            "Cement": inv_cement,
            "Steel": inv_steel,
            "Bricks": inv_bricks,
            "Sand": inv_sand
        }
        
        # --- GENERATE PLAN ---
        st.markdown("---")
        if st.button("🚀 GENERATE PROCUREMENT PLAN"):
            with st.spinner("Optimizing Cost & Logistics..."):
                df_plan = optimize_resources.optimize_procurement(preds)
                
                # METRICS DISPLAY
                total_cost = df_plan['Est_Cost'].sum()
                st.dataframe(df_plan)
                
                c1, c2 = st.columns(2)
                c1.metric("💰 Total Estimated Cost", f"INR {total_cost:,.2f}")
                
                # Check for "Express" or "Premium" usage
                has_premium = df_plan['Note'].str.contains('⚠️').any()
                if has_premium:
                    c2.error("⚠️ Plan includes Express/Premium sourcing due to shortages!")
                else:
                    c2.success("✅ Standard Sourcing Sufficient.")
                
                # PDF REPORT
                st.markdown("### 📄 Export Report")
                try:
                    pdf_bytes = generate_pdf_report(df_plan, total_cost, preds.get('risk_class', 'Unknown'), proj_id, week_num, preds)
                    st.download_button(
                        label="⬇️ Download Procurement Invoice (PDF)",
                        data=pdf_bytes,
                        file_name=f"Procurement_Plan_{proj_id}_Week{week_num}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF Generation Failed: {e}")
                
# =========================================================
# PAGE 3: ANALYTICS DASHBOARD
# =========================================================
elif page == "📈 Analytics Dashboard":
    st.title("📈 Project Analytics")
    
    # Load Data
    project_root = os.path.dirname(backend_path)
    log_file = os.path.join(project_root, 'outputs', 'pipeline_results', 'prediction_history.csv')
    
    if not os.path.exists(log_file):
        st.warning("No historical data found. Run some predictions first!")
    else:
        df_log = pd.read_csv(log_file)
        
        # Filter by Project ID
        df_proj = df_log[df_log['Project_ID'] == proj_id]
        
        if df_proj.empty:
            st.warning(f"No records for Project ID: {proj_id}")
        else:
            # METRICS ROW
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Weeks Logged", len(df_proj))
            avg_risk = df_proj['risk_score'].mean()
            m2.metric("Avg Risk Score", f"{avg_risk:.2f}/15")
            last_gap = df_proj.iloc[-1].get('Progress_Gap_Pct', 0) if 'Progress_Gap_Pct' in df_proj.columns else 0
            m3.metric("Current Schedule Gap", f"{last_gap}%", delta_color="inverse")
            
            st.markdown("---")
            
            # CHARTS ROW 1
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("📅 Schedule Tracking (Planned vs Actual)")
                if 'Progress_Gap_Pct' in df_proj.columns:
                    st.line_chart(df_proj.set_index('Week_Number')['Progress_Gap_Pct'])
                    st.caption("Positive values indicate delays (Actual < Planned).")
                else:
                    st.warning("Progress data not available in older logs.")
                    
            with c2:
                st.subheader("🧱 Material Demand Trend")
                mat_cols = ['req_cement_bags', 'req_steel_kg', 'req_bricks_nos']
                # clean column names for display
                df_mat = df_proj.set_index('Week_Number')[mat_cols]
                df_mat.columns = ["Cement", "Steel", "Bricks"]
                st.bar_chart(df_mat)
            
            st.markdown("---")
            
            # CHARTS ROW 2
            st.subheader("👷 Labour Workforce Trend")
            lab_cols = ['req_skilled_labour', 'req_unskilled_labour']
            df_lab = df_proj.set_index('Week_Number')[lab_cols]
            df_lab.columns = ["Skilled Masons", "Unskilled Helpers"]
            st.line_chart(df_lab)
            
            # Raw Data
            with st.expander("View Raw Data Log"):
                st.dataframe(df_proj)
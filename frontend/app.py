import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from fpdf import FPDF
import base64

# ... (Imports)

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

# ... (Page 1 Logic) ...

# =========================================================
# PAGE 2: OPTIMIZATION ENGINE
# =========================================================
elif page == "⚙️ Optimization Engine":
    st.title("⚙️ Resource Optimization Engine")
    # ... (Optimization Logic)
    
        if st.button("🚀 GENERATE PROCUREMENT PLAN"):
            # ... (Optimization Execution) ...
            
                # METRICS DISPLAY
                # ...
                
                # PDF REPORT
                st.markdown("### 📄 Export Report")
                pdf_bytes = generate_pdf_report(df_plan, total_cost, preds.get('risk_class', 'Unknown'), proj_id, week_num, preds)
                st.download_button(
                    label="⬇️ Download Procurement Invoice (PDF)",
                    data=pdf_bytes,
                    file_name=f"Procurement_Plan_{proj_id}_Week{week_num}.pdf",
                    mime="application/pdf"
                )
                
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
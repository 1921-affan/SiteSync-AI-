import pandas as pd
import numpy as np
import random
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# paths relative to where this script is run (assumed backend/scripts)
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/raw')
INPUT_MATERIAL_FILE = os.path.join(DATA_DIR, "train_materials_final.csv")
INPUT_LABOUR_FILE = os.path.join(DATA_DIR, "train_labour_final.csv")
OUTPUT_RISK_FILE = os.path.join(DATA_DIR, "train_risk_final.csv")

# ==========================================
# 2. GENERATE RISK DATA (Logic-Driven Simulation)
# ==========================================
def generate_risk(df_mat, df_lab):
    print("⚠️ Generating Delay Risk Data with Enhanced Logic (Linked to Labour)...")

    # --- PRE-PROCESSING: Aggregate Labour Data by Project & Week ---
    # The labour file has multiple rows per week (different tasks).
    # We need the TOTAL labour demand for the week to assess risk.
    print("   Aggregating Labour Demand...")
    lab_agg = df_lab.groupby(['project_id', 'week_number']).agg({
        'req_skilled_labour': 'sum',
        'req_unskilled_labour': 'sum'
    }).reset_index()
    
    # Merge Labour info into Material (Main Timeline) Data
    # Use left join because some weeks might have material usage but no labour (rare, but possible)
    df_merged = pd.merge(df_mat, lab_agg, on=['project_id', 'week_number'], how='left')
    df_merged['req_skilled_labour'] = df_merged['req_skilled_labour'].fillna(0)
    df_merged['req_unskilled_labour'] = df_merged['req_unskilled_labour'].fillna(0)

    # Group by Project to define the timeline
    project_meta = df_merged.groupby('project_id').agg(
        total_weeks=('week_number', 'max'),
        total_floors=('total_floors', 'first'),
        property_type=('property_type', 'first')
    ).reset_index()

    risk_rows = []

    # Iterate through each project
    for _, proj in project_meta.iterrows():
        pid = proj['project_id']
        tot_weeks = proj['total_weeks']

        # --- INITIALIZE SIMULATION STATE ---
        curr_planned = 0.0
        curr_actual = 0.0

        # History Trackers
        cum_mat_delays = 0
        cum_weather_days = 0

        # Ideal Weekly Progress (Linear Baseline)
        weekly_plan_rate = 100.0 / tot_weeks

        # Get project rows sorted by week
        proj_rows = df_merged[df_merged['project_id'] == pid].sort_values('week_number')

        for _, row in proj_rows.iterrows():
            week = row['week_number']
            stage = row['construction_stage']

            # Input Features
            weather = row.get('weather_condition', 'Normal')
            req_skilled = row['req_skilled_labour']
            req_unskilled = row['req_unskilled_labour']
            total_labour_req = req_skilled + req_unskilled

            # --- A. UPDATE PLANNED PROGRESS ---
            curr_planned += weekly_plan_rate
            if curr_planned > 100: curr_planned = 100.0

            # --- B. SIMULATE CONDITIONS (The "Causes") ---

            # 1. Weather
            is_rain = 1 if weather in ["Light_Rain", "Heavy_Rain"] else 0
            if is_rain: cum_weather_days += 1

            # 2. Material Status (Still Random as we don't have stock data)
            rand_mat = random.random()
            if rand_mat < 0.10: mat_status = "Shortage"
            elif rand_mat < 0.30: mat_status = "Limited"
            else: mat_status = "Available"

            delay_prob = 0.6 if mat_status == "Shortage" else 0.05
            mat_delay = 1 if random.random() < delay_prob else 0
            if mat_delay: cum_mat_delays += 1

            # 3. Labour Status (NOW DRIVEN BY DATA)
            # Simulate a "Market Availability Factor" (e.g., 80% to 110% of need available)
            market_supply_factor = np.random.uniform(0.7, 1.1) 
            
            # Additional random crunch events (festivals, strikes)
            if random.random() < 0.05: 
                market_supply_factor *= 0.6  # Sudden drop
                
            available_labour = int(total_labour_req * market_supply_factor)
            labour_shortfall = max(0, int(total_labour_req - available_labour))
            
            # Determine Status based on Shortfall Ratio
            if total_labour_req > 0:
                shortfall_ratio = labour_shortfall / total_labour_req
                if shortfall_ratio > 0.30: lab_status = "Critical"
                elif shortfall_ratio > 0.10: lab_status = "Moderate"
                else: lab_status = "Sufficient"
            else:
                lab_status = "Sufficient" # No demand, so sufficient

            # --- C. CALCULATE ACTUAL PROGRESS (The Consequence) ---
            efficiency = 1.0

            # Apply Penalties
            if weather == "Heavy_Rain": efficiency -= 0.8
            elif weather == "Light_Rain": efficiency -= 0.3
            elif weather == "Heatwave": efficiency -= 0.2
            elif weather == "High_Wind": efficiency -= 0.15

            if mat_status == "Shortage": efficiency -= 0.5
            if mat_status == "Limited": efficiency -= 0.2
            
            # Labour Penalty (Proportional to shortfall)
            if total_labour_req > 0:
                labour_penalty = (labour_shortfall / total_labour_req) * 1.5 # Impact magnified
                efficiency -= min(0.6, labour_penalty) # Cap max penalty
            
            if mat_delay: efficiency -= 0.25

            efficiency = max(0.0, efficiency)

            # Update Actuals
            gain = weekly_plan_rate * efficiency
            curr_actual += gain
            if curr_actual > 100: curr_actual = 100.0

            # --- D. RISK SCORING ---
            score = 0
            progress_gap = curr_planned - curr_actual

            if progress_gap >= 30: score += 3
            elif progress_gap >= 20: score += 2
            elif progress_gap >= 10: score += 1

            if weather == "Heavy_Rain": score += 2
            elif weather in ["Light_Rain", "High_Wind", "Heatwave"]: score += 1

            if mat_status == "Shortage": score += 2
            elif mat_status == "Limited": score += 1
            if mat_delay == 1: score += 1
            if cum_mat_delays >= 3: score += 2

            if lab_status == "Critical": score += 2
            elif lab_status == "Moderate": score += 1
            if labour_shortfall >= 10: score += 2
            elif labour_shortfall >= 5: score += 1

            if stage == 3 and score >= 2: score += 1
            if stage == 5 and weather in ["Heavy_Rain", "Light_Rain"]: score += 1
            if week > (0.7 * tot_weeks) and progress_gap >= 15: score += 1

            if score <= 2: final_risk = 0      # Low
            elif 3 <= score <= 5: final_risk = 1 # Medium
            else: final_risk = 2               # High

            # --- E. STORE ---
            risk_rows.append([
                pid, week, stage,
                round(curr_planned, 2),
                round(curr_actual, 2),
                round(progress_gap, 2),
                mat_status,
                lab_status,
                weather,
                is_rain,
                mat_delay,
                cum_mat_delays,
                labour_shortfall,
                score,
                final_risk
            ])

    cols = [
        "project_id", "week_number", "construction_stage",
        "planned_progress_pct", "actual_progress_pct",
        "progress_gap_pct",
        "material_avail_status", "labour_avail_status",
        "weather_condition", "weather_impact_flag", "material_delay_flag",
        "hist_material_delay_count",
        "labour_shortfall_est",
        "risk_score_debug",
        "delay_risk"
    ]

    return pd.DataFrame(risk_rows, columns=cols)

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        if not os.path.exists(INPUT_MATERIAL_FILE):
             print(f"❌ Error: {INPUT_MATERIAL_FILE} not found.")
             exit()
        if not os.path.exists(INPUT_LABOUR_FILE):
             print(f"❌ Error: {INPUT_LABOUR_FILE} not found (Required for linked logic).")
             exit()

        print(f"📂 Loading Materials: {INPUT_MATERIAL_FILE}")
        df_materials = pd.read_csv(INPUT_MATERIAL_FILE)
        
        print(f"📂 Loading Labour: {INPUT_LABOUR_FILE}")
        df_labour = pd.read_csv(INPUT_LABOUR_FILE)

        # Generate
        df_risk = generate_risk(df_materials, df_labour)

        # Save
        print(f"💾 Saving {len(df_risk)} rows to {OUTPUT_RISK_FILE}...")
        df_risk.to_csv(OUTPUT_RISK_FILE, index=False)
        print("✅ Done. Risk dataset generated with Labour dependencies.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

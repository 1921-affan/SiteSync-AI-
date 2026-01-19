import pandas as pd
import pulp
import os
import sys

# ==========================================
# 1. SETUP & PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Data/Static is relative to this script
STATIC_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'static')
MAT_COST_FILE = os.path.join(STATIC_DIR, "material_costs.csv")
LAB_COST_FILE = os.path.join(STATIC_DIR, "labour_costs.csv")

def optimize_procurement(predictions):
    """
    Runs PuLP Optimization to generate a procurement plan.
    
    predictions (dict):
        - req_cement_bags, req_steel_kg, etc.
        - req_skilled_labour, req_unskilled_labour
        - risk_class (Low/Medium/High)
        - project_id (Optional)
        
    Returns:
        pd.DataFrame: The optimized plan.
    """
    
    # ==========================================
    # 2. LOAD STATIC DATA
    # ==========================================
    if not os.path.exists(MAT_COST_FILE) or not os.path.exists(LAB_COST_FILE):
        raise FileNotFoundError(f"Missing Static Cost Files in {STATIC_DIR}")

    df_mat_costs = pd.read_csv(MAT_COST_FILE)
    df_lab_costs = pd.read_csv(LAB_COST_FILE)

    # ==========================================
    # 3. DEFINE SAFETY BUFFERS
    # ==========================================
    risk_class = predictions.get('risk_class', 'Low')
    risk_multipliers = {
        "Low": 1.00, 0: 1.00,   # Buy exactly what's needed
        "Medium": 1.10, 1: 1.10, # +10% Safety Buffer
        "High": 1.20, 2: 1.20,   # +20% Safety Buffer (Panic Mode)
        "Unknown": 1.00
    }
    buffer = risk_multipliers.get(risk_class, 1.0)
    
    # ==========================================
    # 4. INITIALIZE OPTIMIZATION
    # ==========================================
    prob = pulp.LpProblem("Weekly_Resource_Optimization", pulp.LpMinimize)
    
    # ------------------------------------------
    # A. MATERIAL LOGIC
    # ------------------------------------------
    mat_mapping = {
        "req_cement_bags": "Cement",
        "req_steel_kg": "Steel",
        "req_bricks_nos": "Bricks",
        "req_sand_tons": "Sand"
    }
    
    mat_vars = {} 
    total_mat_cost = 0
    
    for pred_col, mat_name in mat_mapping.items():
        if pred_col in predictions:
            # Demand & Buffer
            base_demand = float(predictions[pred_col])
            target_demand = base_demand * buffer
            
            # --- INVENTORY LOGIC ---
            # Subtract Current Inventory if provided
            current_stock = 0.0
            if 'inventory' in predictions:
                 # Map prediction keys to inventory keys (e.g., 'req_cement_bags' -> 'Cement')
                 current_stock = float(predictions['inventory'].get(mat_name, 0.0))
            
            net_demand = max(0.0, target_demand - current_stock)
            
            # If net demand is 0, we don't need to buy anything, but we still might want to track it
            if net_demand == 0:
                 mat_vars[mat_name] = {
                    "reg_var": pulp.LpVariable(f"Qty_{mat_name}_Reg_Zero", lowBound=0, upBound=0), 
                    "exp_var": pulp.LpVariable(f"Qty_{mat_name}_Exp_Zero", lowBound=0, upBound=0),
                    "target": target_demand, "unit": "N/A", "reg_cost": 0, "exp_cost": 0,
                    "stock_used": current_stock # Track this for reporting
                }
                 continue

            # Cost Info
            cost_row = df_mat_costs[df_mat_costs['material_name'] == mat_name]
            if cost_row.empty: continue
            cost_info = cost_row.iloc[0]
            
            # Variables (Regular vs Express)
            qty_reg = pulp.LpVariable(f"Qty_{mat_name}_Regular", lowBound=0, cat='Continuous')
            qty_exp = pulp.LpVariable(f"Qty_{mat_name}_Express", lowBound=0, cat='Continuous')
            
            # Constraints
            # 1. Physical Need (Net Demand)
            prob += (qty_reg + qty_exp >= net_demand)
            # 2. Supply Limit
            prob += (qty_reg <= cost_info['max_supply_weekly_regular'])
            
            # Objective
            total_mat_cost += (qty_reg * cost_info['cost_regular_inr']) + \
                              (qty_exp * cost_info['cost_express_inr'])
            
            mat_vars[mat_name] = {
                "reg_var": qty_reg, "exp_var": qty_exp, 
                "target": net_demand, "unit": cost_info['unit'], # Target is now NET demand
                "reg_cost": cost_info['cost_regular_inr'],
                "exp_cost": cost_info['cost_express_inr'],
                "stock_used": current_stock,
                "gross_demand": target_demand
            }

    # ------------------------------------------
    # B. LABOUR LOGIC
    # ------------------------------------------
    # Dictionary keys from predict_labour.py are: req_skilled_labour, req_unskilled_labour
    lab_mapping = {
        "req_skilled_labour": "Skilled_Mason",
        "req_unskilled_labour": "Unskilled_Helper"
    }

    lab_vars = {}
    total_lab_cost = 0

    for pred_col, role_name in lab_mapping.items():
        if pred_col in predictions:
            # Demand (People)
            base_workers = float(predictions[pred_col])
            target_workers = base_workers * buffer 
            
            cost_row = df_lab_costs[df_lab_costs['labour_role'] == role_name]
            if cost_row.empty: continue
            cost_info = cost_row.iloc[0]
            
            # Variables (Regular vs Premium/Overtime)
            hire_reg = pulp.LpVariable(f"Hire_{role_name}_Regular", lowBound=0, cat='Integer')
            hire_prem = pulp.LpVariable(f"Hire_{role_name}_Premium", lowBound=0, cat='Integer')
            
            # Constraints
            prob += (hire_reg + hire_prem >= target_workers)
            prob += (hire_reg <= cost_info['max_regular_supply'])
            
            # Objective (Cost per Week = 6 days)
            weekly_reg = cost_info['daily_wage_inr'] * 6
            weekly_prem = (cost_info['daily_wage_inr'] * 1.5) * 6
            
            total_lab_cost += (hire_reg * weekly_reg) + (hire_prem * weekly_prem)
            
            lab_vars[role_name] = {
                "reg_var": hire_reg, "prem_var": hire_prem,
                "target": target_workers
            }

    # ------------------------------------------
    # C. SOLVE
    # ------------------------------------------
    prob += total_mat_cost + total_lab_cost
    prob.solve()
    
    # ------------------------------------------
    # D. FORMAT OUTPUT
    # ------------------------------------------
    output_rows = []
    
    # Materials
    for name, data in mat_vars.items():
        # SAFELY GET VALUES (Handle None from Solver)
        val_reg = data['reg_var'].varValue
        val_exp = data['exp_var'].varValue
        
        q_reg = val_reg if val_reg is not None else 0.0
        q_exp = val_exp if val_exp is not None else 0.0
        
        cost = (q_reg * data['reg_cost']) + (q_exp * data['exp_cost'])
        
        note = "Standard"
        if q_exp > 0: note = "⚠️ Express Delivery"
        
        output_rows.append({
            "Category": "Material", "Resource": name, 
            "Total_Need_With_Buffer": round(data.get('gross_demand', data['target']), 1),
            "Inventory_Used": round(data.get('stock_used', 0), 1),
            "Net_Order_Qty": round(data['target'], 1),
            "Unit": data['unit'], "Sourced_Regular": round(q_reg, 1), 
            "Sourced_Premium": round(q_exp, 1), "Est_Cost": round(cost, 2), "Note": note
        })
        
    # Labour
    for name, data in lab_vars.items():
        val_reg = data['reg_var'].varValue
        val_prem = data['prem_var'].varValue
        
        h_reg = val_reg if val_reg is not None else 0.0
        h_prem = val_prem if val_prem is not None else 0.0
        
        # Recalculate cost for display
        c_reg = df_lab_costs[df_lab_costs['labour_role']==name].iloc[0]['daily_wage_inr'] * 6
        c_prem = c_reg * 1.5
        cost = (h_reg * c_reg) + (h_prem * c_prem)
        
        note = "Standard"
        if h_prem > 0: note = "⚠️ Overtime/Agency"
        
        output_rows.append({
            "Category": "Labour", "Resource": name, "Target_Qty": round(data['target'], 1),
            "Unit": "People", "Sourced_Regular": round(h_reg, 1), 
            "Sourced_Premium": round(h_prem, 1), "Est_Cost": round(cost, 2), "Note": note
        })
        
    return pd.DataFrame(output_rows)

if __name__ == "__main__":
    # Test
    test_pred = {
        "req_cement_bags": 500, "req_steel_kg": 2000, 
        "req_bricks_nos": 5000, "req_sand_tons": 20,
        "req_skilled_labour": 15, "req_unskilled_labour": 40,
        "risk_class": "High"
    }
    print(optimize_procurement(test_pred))

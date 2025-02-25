import pandas as pd
import numpy as np
import random



from bokeh.layouts import row, column
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    WheelZoomTool,
    TapTool,
    CategoricalColorMapper,
    LegendItem,
    Select,
)
from bokeh.events import Tap
from functools import partial
from tornado import gen
from threading import Thread

import sys
import os

sys.path.append(os.path.relpath("../"))
from rilacs.incremental_audit import *


genders = ['Female', 'Male']
races = ['Asian', 'Black', 'Hispanic', 'MultiRace', 'Native', 'PacificIslander', 'White']
race_gender_combinations = ['Asian_Female', 'Asian_Male', 'Black_Female', 'Black_Male', 'Hispanic_Female', 'Hispanic_Male', 
 'MultiRace_Female', 'MultiRace_Male', 'Native_Female', 'Native_Male', 'PacificIslander_Female', 
 'PacificIslander_Male', 'White_Female', 'White_Male']

file_path = "./Employment Audit  - NYC 144 Audit - All.csv"
df = pd.read_csv(file_path)
company_data = df.iloc[0]

def get_column_values(df, category, metric):
    column_name = f"{category}_{metric}"
    return df[column_name] 

# Function to build a dictionary for a given list of categories
def build_category_dict(categories, df):
    category_dict = {}
    for category in categories:
        category_dict[category] = {
            "nApplicants": get_column_values(df, category, "nApplicants"),
            "nSelected": get_column_values(df, category, "nSelected"),
            "SelectionRate": get_column_values(df, category, "SelectionRate"),
            "ImpactRatio": get_column_values(df, category, "ImpactRatio")
        }
    return category_dict

# Build dictionaries
gender_data = build_category_dict(genders, company_data)
race_data = build_category_dict(races, company_data)
race_gender_data = build_category_dict(race_gender_combinations, company_data)
print(gender_data)


def generate_candidates(data_dict, data_type="gender"):
    candidates = []

    for category, metrics in data_dict.items():
    
        n_applicants = int(metrics.get("nApplicants", 0).replace(',', ''))  
        n_selected = int(metrics.get("nSelected", 0).replace(',', ''))  

        # Determine gender and race from category name
        if data_type == "gender":
            gender, race = category, "N/A"
        elif data_type == "race":
            gender, race = "N/A", category
        elif data_type == "race-gender":
            race,gender= category.split("_")  # Assumes format "Race_Gender"
        else:
            raise ValueError("Invalid data_type. Choose from 'gender', 'race', or 'race-gender'.")

        race_gender = f"{race}_{gender}" if race != "N/A" and gender != "N/A" else "N/A"

        # Append candidates with gender, race, race-gender, and selection status
        candidates.extend([{"Gender": gender, "Race": race, "Race_Gender": race_gender, "Selected": False}] * n_applicants)
        candidates.extend([{"Gender": gender, "Race": race, "Race_Gender": race_gender, "Selected": True}] * n_selected)


    return random.sample(candidates, len(candidates) )



# Print a sample of the list for verification

def run_employment_audit(candidates, selected_audit_method="Kelly", group_A=None, group_B=None, audit_type="Gender", ref_group ="A", rho=4/5):
    if not group_A or not group_B:
        raise ValueError("You must specify two groups (group_A and group_B) for the audit.")
    

    if ref_group not in ["A", "B"]:
        raise ValueError("Invalid reference_group. Choose 'A' or 'B'.")


    ballot_dict = {}  # Store selection statuses
    audits = {}  # Store audit processes
    
    
    filtered_candidates_A, filtered_candidates_B =[],[]

    # Filter candidates based on the audit type
    if audit_type == "Gender":
        filtered_candidates_A = [c for c in candidates if c["Gender"] == group_A]
        filtered_candidates_B = [c for c in candidates if c["Gender"] == group_B]
    elif audit_type == "Race":
        filtered_candidates_A = [c for c in candidates if c["Race"] == group_A]
        filtered_candidates_B = [c for c in candidates if c["Race"] == group_B]
    elif audit_type == "Race_Gender":
        filtered_candidates_A = [c for c in candidates if f"{c['Race_Gender']}" == group_A]
        filtered_candidates_B = [c for c in candidates if f"{c['Race_Gender']}" == group_B]
    
    # Count selected vs not selected for each group
    total_selected_A = sum(1 for c in filtered_candidates_A if c["Selected"])
    total_not_selected_A = sum(1 for c in filtered_candidates_A if not c["Selected"])
    n_A = total_selected_A + total_not_selected_A
    
    total_selected_B = sum(1 for c in filtered_candidates_B if c["Selected"])
    total_not_selected_B = sum(1 for c in filtered_candidates_B if not c["Selected"])
    n_B = total_selected_B + total_not_selected_B
    N = n_A + n_B

    ballots = filtered_candidates_A + filtered_candidates_B
    random.shuffle(ballots)

    audit_methods_dict = {
        "Kelly": lambda n_A, n_B: Betting_Audit(
            bettor=Kelly_Bettor(n_A=n_A, n_B=n_B), N=len(ballots)
        ),
        "SqKelly": lambda n_A, n_B: Betting_Audit(
            bettor=DistKelly_Bettor(), N=len(ballots)
        ),
        "Hoeffding": lambda n_A, n_B: Hoeffding_Audit(N=len(ballots)),
    }

    if (total_selected_A + total_not_selected_A == 0) or (total_selected_B + total_not_selected_B == 0):
        print(f"No candidates found for groups {group_A} or {group_B}")
        return None

    #print(f"Comparing Groups: {group_A} vs {group_B}")
    #print(f"{group_A} - Selected: {total_selected_A}, Not Selected: {total_not_selected_A}")
    #print(f"{group_B} - Selected: {total_selected_B}, Not Selected: {total_not_selected_B}")
    #print("N: ", N)

    # Initialize the audit
    audits["selection"] = audit_methods_dict[selected_audit_method](n_A=n_A, n_B=n_B)

    

   # print("Starting employment selection audit...")


    # Track audit progress
    new_data = {"selection": {"t": [], "l": []}}
    fair = 0
    audit_end_time =0
    #print("N",N)
    #print("len ", len(ballots))
    for i in range(len(ballots)):
        ballot = ballots[i]
        group = ballot[audit_type]

       
        audit_update_value =0
        if ballot["Selected"]:
            if group == group_A and ref_group == "A":
                audit_update_value = - rho* N / n_A
            elif group == group_A and ref_group == "B":
                audit_update_value =  N / n_A
            elif group == group_B  and ref_group == "A":
                audit_update_value = N / n_B
            elif group == group_B and ref_group == "B":
                audit_update_value = -rho* N / n_B
        else:
            audit_update_value =0

        #print(audit_update_value)

         # Stop early if audit conditions are met
        audit_end_time = len(candidates)
        eta = 0.001
        #print(audit_update_value)
        if (audits["selection"].l >eta):
            #print("Stopping time =", i + 1)
            fair = 1
            audit_end_time = i
            break
      #  if i % 2000 == 0:
           # print("Lower Bound = ", audits["selection"].l)

        # Update the audit with the new ballot
        l = audits["selection"].update_cs(audit_update_value)
        new_data["selection"]["t"].append(audits["selection"].t)
        new_data["selection"]["l"].append(l)
   # print("fair: ", fair)
    return audit_end_time, fair, new_data  # Return audit tracking data

gender_candidates = (generate_candidates(gender_data, data_type="gender"))
race_candidates =(generate_candidates(race_data, data_type="race"))
race_gender_candidates = (generate_candidates(race_gender_data, data_type="race-gender"))
#print(race_gender_candidates[:10])

# Run an audit comparing "Female" vs "Male"
type = ["Gender", "Race", "Race_Gender"]
#end_time_gender, innocent_gender, audit_results_gender = run_employment_audit(gender_candidates, selected_audit_method="Kelly", group_A="Female", group_B="Male", 
 #   audit_type=type[0], ref_group="B")


#end_time_race_gender, innocent_race_gender, audit_results_race_gender = run_employment_audit(race_gender_candidates, selected_audit_method="Kelly", group_A="Asian_Female", group_B="Asian_Male", 
   # audit_type=type[2], ref_group="A")

#end_time_race, innocent_race, audit_results_race= run_employment_audit(race_candidates, selected_audit_method="Kelly", group_A="Asian", group_B="Native", 
   # audit_type=type[1], ref_group="B")





def check_fairness_impact_ratio(group_A_data, group_B_data, reference_group="A", rho =0.8):
   
    # Extract selection rates
    selection_rate_A = float(group_A_data.get("SelectionRate", 0).replace('%', ''))
    selection_rate_B = float(group_B_data.get("SelectionRate", 0).replace('%', ''))

    # Compute Impact Ratio in the correct direction
    if reference_group == "B":
        impact_ratio = selection_rate_A / selection_rate_B if selection_rate_B > 0 else 0
    else:
        impact_ratio = selection_rate_B / selection_rate_A if selection_rate_A > 0 else 0
    #print(impact_ratio)
    # Check if the impact ratio meets the 4/5 rule
    return 1 if impact_ratio >= rho else 0  # 1 = Fair, 0 = Unfair


def run_multiple_audits(
    candidates, 
    data_dict,
    selected_audit_method="Kelly", 
    group_A=None, 
    group_B=None, 
    audit_type="gender", 
    num_trials=100
):
    
    total_time_A = 0
    correct_audit_count_A = 0  
    total_time_B = 0
    correct_audit_count_B = 0  
    correct_audit_count_overall = 0  

    # Get fairness based on the 4/5 Impact Ratio rule
    fairness_A= check_fairness_impact_ratio(data_dict[group_A], data_dict[group_B], "A" ,rho=0.8)
    fairness_B= check_fairness_impact_ratio(data_dict[group_A], data_dict[group_B], "B" ,rho=0.8)
    fairness_overall = (fairness_A==1 and fairness_B ==1 )

    for _ in range(num_trials):
        result_A = run_employment_audit(
            candidates, selected_audit_method, group_A, group_B, audit_type, "A")

        result_B = run_employment_audit(
            candidates, selected_audit_method, group_A, group_B, audit_type, "B")

        end_time_A, fairness_audit_A, _ = result_A
        total_time_A += end_time_A
        # Check if the audit result matches the actual fairness
        if fairness_audit_A == fairness_A:
            correct_audit_count_A += 1
    
        end_time_B, fairness_audit_B, _ = result_B
        #print("fair B" ,fairness_audit_B)
        total_time_B += end_time_B
        # Check if the audit result matches the actual fairness
        if fairness_audit_B == fairness_B:
            correct_audit_count_B += 1

        if (fairness_audit_B and fairness_audit_A)== fairness_overall:
            correct_audit_count_overall += 1

    # Compute statistics
    avg_end_time_A = total_time_A / num_trials
    audit_correctness_A = (correct_audit_count_A / num_trials) * 100  
    avg_end_time_B= total_time_B / num_trials
    audit_correctness_B= (correct_audit_count_B/ num_trials) * 100  
    audit_correctness_overall = (correct_audit_count_overall / num_trials) * 100  

    return {
        "Audit Type": group_A +" vs " + group_B,
        "num_candidates":len(candidates),
        "fair_" + group_A +"_as_ref": fairness_A,
        "average_end_time_" + group_A +"_as_ref": avg_end_time_A,
        "audit_correctness_"+ group_A +"_as_ref": audit_correctness_A,
        "fair_" + group_B +"_as_ref": fairness_B,
        "average_end_time_" + group_B +"_as_ref": avg_end_time_B,
        "audit_correctness_"+ group_B +"_as_ref": audit_correctness_B,
        "fair_overall": fairness_overall,
        "audit_correctness_overall": audit_correctness_overall,
    }


audit_stats_gender = run_multiple_audits(
    gender_candidates, gender_data, selected_audit_method="Kelly", group_A="Female", group_B="Male", audit_type="Gender", num_trials=100)


for keys,values in audit_stats_gender.items():
    print(keys + ": " , values)

cigna_data = df.iloc[9]
#print(cigna_data)
gender_data_cigna = build_category_dict(genders, cigna_data)
gender_candidates_Cigna = generate_candidates(gender_data_cigna, data_type="gender")
audit_stats_gender_cigna = run_multiple_audits(
    gender_candidates_Cigna, gender_data_cigna, selected_audit_method="Kelly", group_A="Female", group_B="Male", audit_type="Gender", num_trials=100)

for keys,values in audit_stats_gender_cigna.items():
    print(keys + ": " , values)

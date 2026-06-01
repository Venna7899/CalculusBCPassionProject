import pandas as pd
import math
import difflib

# Hardcoded League Averages, 75 anchor
AVG_LEAGUE_ATTACK = 75.0
AVG_LEAGUE_DEFENSE = 75.0

# Base goals for poisson scaling
BASE_HOME_GOALS = 1.50
BASE_AWAY_GOALS = 1.10
BASE_NEUTRAL_GOALS = 1.30

# Load Database
print("Loading EA FC 26 Database...")
df = pd.read_csv('EAFC26-Men.csv')
# Clean column names just in case of trailing spaces
df.columns = df.columns.str.strip()
player_names = df['Name'].dropna().tolist()

def get_player_stats(name):
    #Finds the closest matching player name using fuzzy logic.
    name = name.strip()
    matches = difflib.get_close_matches(name, player_names, n=1, cutoff=0.6)
    
    if matches:
        matched_name = matches[0]
        # Return the first row that matches this name
        player_data = df[df['Name'] == matched_name].iloc[0]
        return player_data
    else:
        print(f"  [!] Could not find a match for '{name}'. Assigning base 75 rating.")
        return {'Name': 'Player not found', 'OVR': 75.0}

def calc_category_avg(player_list, category):
    #Calculates the raw average for a positional group.
    total_score = 0
    valid_players = 0
    
    for name in player_list:
        if not name.strip(): continue
        
        player = get_player_stats(name)
        if player is not None:
            print(f"  -> Found: {player['Name']} (OVR: {player['OVR']})")
            
            # Extract attributes safely, defaulting to overall if missing
            try:
                if player['Name'] == 'Player not found':
                    score = float(player['OVR'])
                else:
                    if category == 'Offense':
                        score = (float(player['SHO']) + float(player['Positioning']) + float(player['Finishing'])) / 3
                    elif category == 'Midfield':
                        score = (float(player['PAS']) + float(player['Vision']) + float(player['Dribbling'])) / 3
                    elif category == 'Defense':
                        # Assuming defensive inputs cover GK duties for the prototype
                        score = (float(player['DEF']) + float(player['PHY']) + float(player['Interceptions'])) / 3
                    elif category == 'Goalie':
                        score = (float(player['GK Diving']) + float(player['GK Handling']) + float(player['GK Positioning']) + float(player['GK Reflexes'])) / 4
                    
                    if math.isnan(score):
                        score = float(player['OVR'])
                
                total_score += score
                valid_players += 1
            except (ValueError, TypeError):
                # Fallback to overall rating if specific stats are missing/corrupted
                total_score += float(player['OVR'])
                valid_players += 1
                
    if valid_players == 0:
        return 75.0 # Fallback to average if no valid players found
    return total_score / valid_players

def poisson_probability(lmbda, k):
    #Calculates Poisson probability for k events given lambda.
    return ((lmbda ** k) * math.exp(-lmbda)) / math.factorial(k)

#USER INPUT (Modified to split by comma)
print("\n=== SOCCER MATCH PREDICTOR ===")
#compName = input("Enter the competition name: ")

print("\n--- TEAM 1 ---")
t1Att = input("Enter Team 1 attackers (comma separated): ").split(',')
t1Mid = input("Enter Team 1 midfielders (comma separated): ").split(',')
t1Def = input("Enter Team 1 defenders (comma separated): ").split(',')
t1Gk = input("Enter Team 1 Goalkeeper (comma separated): ").split(',')

print("\n--- TEAM 2 ---")
t2Att = input("Enter Team 2 attackers (comma separated): ").split(',')
t2Mid = input("Enter Team 2 midfielders (comma separated): ").split(',')
t2Def = input("Enter Team 2 defenders (comma separated): ").split(',')
t2Gk = input("Enter Team 2 Goalkeeper (comma separated): ").split(',')

homeAdvantage = input("\nWho is playing home? (team1, team2, neither): ").strip().lower()

#Hidden Layer
print("\nProcessing Team 1 Roster...")
t1_A_raw = calc_category_avg(t1Att, 'Offense')
t1_M_raw = calc_category_avg(t1Mid, 'Midfield')
t1_Gk_raw = calc_category_avg(t1Gk, 'Goalie')
t1_D_raw = calc_category_avg(t1Def, 'Defense') * 0.7 + t1_Gk_raw * 0.3

print("\nProcessing Team 2 Roster...")
t2_A_raw = calc_category_avg(t2Att, 'Offense')
t2_M_raw = calc_category_avg(t2Mid, 'Midfield')
t2_Gk_raw = calc_category_avg(t2Gk, 'Goalie')
t2_D_raw = calc_category_avg(t2Def, 'Defense') * 0.7 + t2_Gk_raw * 0.3

# Midfield Battle
total_midfield = t1_M_raw + t2_M_raw
if total_midfield > 0:
    control_1 = t1_M_raw / total_midfield
else:
    control_1 = 0.5

# Adjusted Scores (Possession boosts attack, mitigates defense)
t1_A_adj = t1_A_raw * (1 + (control_1 - 0.5))
t2_D_adj = t2_D_raw * (1 - (control_1 - 0.5))

t2_A_adj = t2_A_raw * (1 + ((1 - control_1) - 0.5))
t1_D_adj = t1_D_raw * (1 - ((1 - control_1) - 0.5))

# Calculating Lamda xG
t1_base_goals = BASE_NEUTRAL_GOALS
t2_base_goals = BASE_NEUTRAL_GOALS

# Set Home/Away modifiers
if homeAdvantage == 'team1':
    t1_base_goals = BASE_HOME_GOALS
    t2_base_goals = BASE_AWAY_GOALS
elif homeAdvantage == 'team2':
    t1_base_goals = BASE_AWAY_GOALS
    t2_base_goals = BASE_HOME_GOALS

# Lambda Formula: (Att Strength) * (Opp Def Strength) * Base Goals
lambda_1 = (t1_A_adj / AVG_LEAGUE_ATTACK) * (AVG_LEAGUE_DEFENSE / t2_D_adj) * t1_base_goals
lambda_2 = (t2_A_adj / AVG_LEAGUE_ATTACK) * (AVG_LEAGUE_DEFENSE / t1_D_adj) * t2_base_goals

#Generating the probabilities and the grid
MAX_GOALS = 6
win_1_prob = 0.0
win_2_prob = 0.0
draw_prob = 0.0

grid = {}

for g1 in range(MAX_GOALS):
    for g2 in range(MAX_GOALS):
        prob = poisson_probability(lambda_1, g1) * poisson_probability(lambda_2, g2)
        grid[(g1, g2)] = prob
        
        if g1 > g2:
            win_1_prob += prob
        elif g2 > g1:
            win_2_prob += prob
        else:
            draw_prob += prob

#GUI
print(f"\n==============================================")
print(f" MATCH PREDICTION")
print(f"==============================================")
print(f"Team 1 Expected Goals (xG): {lambda_1:.2f}")
print(f"Team 2 Expected Goals (xG): {lambda_2:.2f}")
print(f"Midfield Control: Team 1 ({control_1*100:.1f}%) | Team 2 ({(1-control_1)*100:.1f}%)")
print(f"----------------------------------------------")
print(f"Team 1 Win:  {win_1_prob*100:.1f}%")
print(f"Draw:        {draw_prob*100:.1f}%")
print(f"Team 2 Win:  {win_2_prob*100:.1f}%")
print(f"==============================================\n")

print("SCORELINE PROBABILITY MATRIX (Team 1 Goals \\ Team 2 Goals)\n")
print(f"{'':<6} | {'0':<6} | {'1':<6} | {'2':<6} | {'3':<6} | {'4':<6}")
print("-" * 45)

for r in range(5):
    row_str = f"  {r}    |"
    for c in range(5):
        # Multiply by 100 to show percentage
        val = grid[(r, c)] * 100
        row_str += f" {val:>4.1f}% |"
    print(row_str)

best_scoreline = max(grid, key=grid.get)
highest_percentage = grid[best_scoreline] * 100

home_goals, away_goals = best_scoreline

# Quick conditional to tag the type of result it is
if home_goals > away_goals:
    outcome_tag = "Team 1 Win"
elif away_goals > home_goals:
    outcome_tag = "Team 2 Win"
else:
    outcome_tag = "Draw"

print("\n----------------------------------------------")
print(f"MOST PROBABLE OUTCOME: {home_goals} - {away_goals} ({outcome_tag})")
print(f"Probability: {highest_percentage:.2f}%")
print("----------------------------------------------")
import numpy as np # linear algebra
import os
import pandas as pd
import copy # Needed for deep copying the points table
import random # Needed for Monte Carlo simulation
import time # Optional: to track simulation time
# Load the dataframe (replace with your actual path)
# Assuming the file 'ipl-2025-UTC.csv' exists in the expected path
try:
    # Use the working directory path from the previous step if applicable
    # df_abbreviated = pd.read_csv('/kaggle/working/ipl_2025_abbreviated.csv')
    # Or reload from original and process again:
    df_orig = pd.read_csv('/kaggle/input/ipl2025fixtures/ipl-2025-UTC.csv')
    df_orig = df_orig.drop(columns=['Round Number', 'Date', 'Location'])

    team_abbreviations = {
        'Kolkata Knight Riders': 'KKR', 'Royal Challengers Bengaluru': 'RCB',
        'Sunrisers Hyderabad': 'SRH', 'Rajasthan Royals': 'RR',
        'Chennai Super Kings': 'CSK', 'Mumbai Indians': 'MI',
        'Delhi Capitals': 'DC', 'Lucknow Super Giants': 'LSG',
        'Gujarat Titans': 'GT', 'Punjab Kings': 'PBKS'
    }
    df_abbreviated = df_orig.copy()
    df_abbreviated['Home Team'] = df_abbreviated['Home Team'].map(team_abbreviations).fillna(df_abbreviated['Home Team'])
    df_abbreviated['Away Team'] = df_abbreviated['Away Team'].map(team_abbreviations).fillna(df_abbreviated['Away Team'])

    if 'Result' not in df_abbreviated.columns:
        df_abbreviated['Result'] = ''
    df_abbreviated['Result'] = df_abbreviated['Result'].fillna('')

except FileNotFoundError:
    print("Error: Schedule CSV file not found. Please check the path.")
    # Create a dummy dataframe for demonstration if file not found
    data = {'Match Number': range(1, 71),
            'Home Team': ['CSK', 'DC', 'KKR', 'PBKS', 'SRH', 'RCB', 'LSG', 'GT', 'RR', 'MI'] * 7,
            'Away Team': ['RCB', 'CSK', 'SRH', 'RR', 'KKR', 'PBKS', 'GT', 'MI', 'DC', 'LSG'] * 7,
            'Result': [''] * 70} # Empty results initially
    df_abbreviated = pd.DataFrame(data)
    teams = sorted(list(set(df_abbreviated['Home Team']).union(set(df_abbreviated['Away Team']))))
    print("Using dummy data for demonstration.")


# Define a function to update the result for a given match number
def update_result(df, match_number, winner):
    # Ensure winner abbreviation is used if full name provided
    winner_abbr = team_abbreviations.get(winner, winner)
    if match_number in df['Match Number'].values:
        df.loc[df['Match Number'] == match_number, 'Result'] = winner_abbr
    else:
        print(f"Warning: Match Number {match_number} not found in schedule.")
    return df

# ==============================================================================
# == UPDATE COMPLETED MATCH RESULTS HERE ==
# ==============================================================================
# As matches complete, add/update lines below with the correct winner
# Make sure to use the abbreviations (KKR, RCB, SRH, etc.)

df_abbreviated = update_result(df_abbreviated, match_number=1, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=2, winner='SRH')
df_abbreviated = update_result(df_abbreviated, match_number=3, winner='CSK')
df_abbreviated = update_result(df_abbreviated, match_number=4, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=5, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=6, winner='KKR')
df_abbreviated = update_result(df_abbreviated, match_number=7, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=8, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=9, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=10, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=11, winner='RR')
df_abbreviated = update_result(df_abbreviated, match_number=12, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=13, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=14, winner='GT')
# Add more lines here as results come in, e.g.:
# 
# df_abbreviated = update_result(df_abbreviated, match_number=14, winner='CSK')
# ... and so on

# ==============================================================================

# Clean up: Ensure all team names are strings and remove any NaN/None
home_teams = df_abbreviated['Home Team'].dropna().astype(str)
away_teams = df_abbreviated['Away Team'].dropna().astype(str)
teams = sorted(list(set(home_teams).union(set(away_teams)))) # Ensure it's a list

# --- Calculate Initial Points Table based on CURRENT completed matches ---
completed_matches = df_abbreviated[df_abbreviated['Result'] != ''].copy()

# Initialize a points dictionary (base state)
initial_points_table = {team: {'Matches': 0, 'Wins': 0, 'Losses': 0, 'Points': 0} for team in teams}

# Calculate wins, losses, and points from completed matches
for _, row in completed_matches.iterrows():
    home = row['Home Team']
    away = row['Away Team']
    winner = row['Result']

    if home in initial_points_table: initial_points_table[home]['Matches'] += 1
    if away in initial_points_table: initial_points_table[away]['Matches'] += 1

    if winner == home:
        if home in initial_points_table:
            initial_points_table[home]['Wins'] += 1
            initial_points_table[home]['Points'] += 2
        if away in initial_points_table: initial_points_table[away]['Losses'] += 1
    elif winner == away: # Check if winner is away team
        if away in initial_points_table:
            initial_points_table[away]['Wins'] += 1
            initial_points_table[away]['Points'] += 2
        if home in initial_points_table: initial_points_table[home]['Losses'] += 1
    # Ignore draws or no results for points calculation

# --- Display Current Standings ---
print("=" * 50)
print(f"Current Standings ({pd.Timestamp.now(tz='Europe/London').strftime('%Y-%m-%d %H:%M %Z')})")
print(f"Based on {len(completed_matches)} completed matches:")
initial_points_df = pd.DataFrame.from_dict(initial_points_table, orient='index')
initial_points_df = initial_points_df.reset_index().rename(columns={'index': 'Team'})
print(initial_points_df.sort_values(by=['Points', 'Wins'], ascending=[False, False]).to_string(index=False))
print("=" * 50)


# --- Monte Carlo Simulation Part ---

# Get remaining matches
remaining_matches = df_abbreviated[df_abbreviated['Result'] == ''].copy()
num_remaining_matches = len(remaining_matches)

if num_remaining_matches == 0:
    print("No remaining matches to simulate. Final standings are above.")
else:
    # --- Monte Carlo Settings ---
    # Increase simulations for better accuracy, decrease if it runs too slow
    num_simulations = 500000
    print(f"Running Monte Carlo simulation with {num_simulations:,} scenarios...")
    print(f"Simulating {num_remaining_matches} remaining matches.")
    start_time = time.time() # Track simulation duration

    # Initialize qualification counters
    top_4_qualifications = {team: 0 for team in teams}
    # Optional: track total points distribution
    # total_points_distribution = {team: [] for team in teams}

    # --- Main Simulation Loop ---
    for i in range(num_simulations):
        # Start with a deep copy of the initial points table for this run
        current_scenario_points = copy.deepcopy(initial_points_table)

        # Simulate each remaining match with a random outcome
        for _, match_info in remaining_matches.iterrows():
            home = match_info['Home Team']
            away = match_info['Away Team']

            # --- Randomly determine winner (50/50 chance) ---
            # TODO: Implement weighted probabilities later if desired
            # e.g., based on current points, historical data, etc.
            winner = random.choice([home, away])

            # Update points, wins, losses, matches for this run
            current_scenario_points[home]['Matches'] += 1
            current_scenario_points[away]['Matches'] += 1

            if winner == home:
                current_scenario_points[home]['Wins'] += 1
                current_scenario_points[home]['Points'] += 2
                current_scenario_points[away]['Losses'] += 1
            else: # Winner is away team
                current_scenario_points[away]['Wins'] += 1
                current_scenario_points[away]['Points'] += 2
                current_scenario_points[home]['Losses'] += 1

        # Scenario complete for this run, determine final standings
        final_standings_df = pd.DataFrame.from_dict(current_scenario_points, orient='index')
        # Sort by Points, then Wins (basic tie-breaker, ignores NRR)
        final_standings_df = final_standings_df.sort_values(by=['Points', 'Wins'], ascending=[False, False])
        final_standings_df = final_standings_df.reset_index().rename(columns={'index': 'Team'})

        # Get the top 4 teams for this scenario
        top_4_teams = final_standings_df['Team'].head(4).tolist()

        # Increment the counter for teams finishing in the top 4
        for team in top_4_teams:
            top_4_qualifications[team] += 1

        # Optional: Store final points for distribution analysis
        # for team in teams:
        #    total_points_distribution[team].append(current_scenario_points[team]['Points'])

        # --- Progress Indicator ---
        if (i + 1) % (num_simulations // 10) == 0: # Update every 10%
            elapsed = time.time() - start_time
            print(f"  ... simulated {i + 1}/{num_simulations} scenarios ({elapsed:.1f} seconds elapsed)")

    # --- Display Results ---
    end_time = time.time()
    print("-" * 50)
    print(f"Monte Carlo Simulation Complete.")
    print(f"Total simulation time: {end_time - start_time:.2f} seconds for {num_simulations:,} scenarios.")
    print("-" * 50)
    print("Estimated Top 4 Qualification Probabilities:")

    results_data = []
    for team in teams:
        count = top_4_qualifications.get(team, 0)
        probability = (count / num_simulations) * 100 if num_simulations > 0 else 0
        results_data.append({
            'Team': team,
            'Top 4 Finishes (Est)': f"{count:,}",
            'Probability (%)': probability
        })

    results_df = pd.DataFrame(results_data)
    # Sort by estimated probability
    results_df = results_df.sort_values(by='Probability (%)', ascending=False)

    # Format for display
    results_df_display = results_df.copy()
    results_df_display['Probability (%)'] = results_df_display['Probability (%)'].map('{:.2f}%'.format)

    print(results_df_display.to_string(index=False))

    print("\nNotes:")
    print(" - Probabilities are estimates based on Monte Carlo simulation.")
    print(" - Assumes a 50/50 chance for each team in every remaining match.")
    print(" - Ranking uses Points, then Wins as tie-breaker (ignores actual NRR).")
    print("=" * 50)

    # Optional: Analyze points distribution (e.g., average points, likely range)
    # print("\nEstimated Final Points Distribution (Average):")
    # points_avg = {team: sum(pts_list)/len(pts_list) if pts_list else 0
    #               for team, pts_list in total_points_distribution.items()}
    # print(pd.Series(points_avg).sort_values(ascending=False).map('{:.1f}'.format))

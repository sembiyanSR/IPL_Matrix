# Required libraries
import numpy as np
import pandas as pd
from itertools import product
from math import factorial, log10
import time
import sys
import os # For checking file paths if needed
import traceback # For printing full tracebacks on error
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For plot aesthetics

# --- Data Loading and Preprocessing ---
try:
    file_path = '/kaggle/input/ipl2025fixtures/ipl-2025-UTC.csv'
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Schedule file not found at: {file_path}")

    df_orig = pd.read_csv(file_path)
    df_orig = df_orig.loc[pd.to_numeric(df_orig['Match Number'], errors='coerce').notna()].copy()
    df_orig['Match Number'] = df_orig['Match Number'].astype(int)
    df_orig = df_orig.drop(columns=['Round Number', 'Date', 'Location'], errors='ignore')
    df_orig = df_orig[:-4] # Remove qualifiers/final
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
    df_abbreviated['Result'] = df_abbreviated['Result'].fillna('').astype(str)

except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    sys.exit("Exiting: Cannot proceed without schedule data.")
except Exception as e:
    print(f"FATAL ERROR: Error loading or processing schedule CSV: {e}")
    traceback.print_exc()
    sys.exit("Exiting: Data processing failed.")

# --- Function to Update Results ---
def update_result(df, match_number, winner):
    winner_abbr = team_abbreviations.get(winner, winner) # Use abbreviation if available
    if match_number in df['Match Number'].values:
        df.loc[df['Match Number'] == match_number, 'Result'] = winner_abbr
    else:
        print(f"Warning: Match Number {match_number} not found in schedule.")
    return df

# ==============================================================================
# == UPDATE COMPLETED MATCH RESULTS HERE ==
# === ADD COMPLETED MATCH RESULTS HERE ===
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
df_abbreviated = update_result(df_abbreviated, match_number=15, winner='KKR')
df_abbreviated = update_result(df_abbreviated, match_number=16, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=17, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=18, winner='RR')
df_abbreviated = update_result(df_abbreviated, match_number=19, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=20, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=21, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=22, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=23, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=24, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=25, winner='KKR')
df_abbreviated = update_result(df_abbreviated, match_number=26, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=27, winner='SRH')
df_abbreviated = update_result(df_abbreviated, match_number=28, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=29, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=30, winner='CSK')
df_abbreviated = update_result(df_abbreviated, match_number=31, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=32, winner='DC')
df_abbreviated = update_result(df_abbreviated, match_number=33, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=34, winner='PBKS')
df_abbreviated = update_result(df_abbreviated, match_number=35, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=36, winner='LSG')
df_abbreviated = update_result(df_abbreviated, match_number=37, winner='RCB')
df_abbreviated = update_result(df_abbreviated, match_number=38, winner='MI')
df_abbreviated = update_result(df_abbreviated, match_number=39, winner='GT')
df_abbreviated = update_result(df_abbreviated, match_number=40, winner='DC')
# ==============================================================================

# --- Prepare Team Data and Initial Standings ---
try:
    home_teams = df_abbreviated['Home Team'].dropna().astype(str)
    away_teams = df_abbreviated['Away Team'].dropna().astype(str)
    teams = sorted(list(set(home_teams).union(set(away_teams))))
    num_teams = len(teams)
    if num_teams == 0:
        raise ValueError("No teams found after processing the schedule.")
    team_to_index = {team: i for i, team in enumerate(teams)}
    index_to_team = {i: team for team, i in team_to_index.items()}

    initial_points_np = np.zeros((num_teams, 4), dtype=np.int32) # M, W, L, Pts

    completed_matches = df_abbreviated[df_abbreviated['Result'] != ''].copy()
    for _, row in completed_matches.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        winner = row['Result'] # Already abbreviated
        home_idx = team_to_index.get(home)
        away_idx = team_to_index.get(away)

        if home_idx is None or away_idx is None:
            print(f"Warning: Skipping completed match {row.get('Match Number', 'N/A')} due to unknown team(s): {home} vs {away}")
            continue

        initial_points_np[home_idx, 0] += 1
        initial_points_np[away_idx, 0] += 1
        if winner == home:
            initial_points_np[home_idx, 1] += 1
            initial_points_np[home_idx, 3] += 2
            initial_points_np[away_idx, 2] += 1
        elif winner == away:
            initial_points_np[away_idx, 1] += 1
            initial_points_np[away_idx, 3] += 2
            initial_points_np[home_idx, 2] += 1
        # No points for ties/no results in this simplified model

except ValueError as e:
    print(f"FATAL ERROR: {e}")
    sys.exit("Exiting: Data validation error.")
except Exception as e:
    print(f"FATAL ERROR: Failed during initial standings calculation: {e}")
    traceback.print_exc()
    sys.exit("Exiting: Could not prepare initial data.")

# --- Display Current Standings ---
print("=" * 50)
print(f"Current Standings ({pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M %Z')})")
print(f"Based on {len(completed_matches)} completed matches:")
initial_points_df_display = pd.DataFrame(
    initial_points_np,
    columns=['Matches', 'Wins', 'Losses', 'Points'],
    index=[index_to_team.get(i, f"Unknown_{i}") for i in range(num_teams)]
)
initial_points_df_display.index.name = 'Team'
print(initial_points_df_display.sort_values(by=['Points', 'Wins'], ascending=[False, False]).to_string())
print("=" * 50)

# --- Prepare Remaining Matches for Simulation ---
try:
    remaining_matches_df = df_abbreviated[df_abbreviated['Result'] == ''].copy()
    num_remaining_matches = len(remaining_matches_df)

    if num_remaining_matches == 0:
        print("No remaining matches to simulate. Final standings are above.")
        sys.exit()

    remaining_matches_info = []
    for _, row in remaining_matches_df.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        home_idx = team_to_index.get(home)
        away_idx = team_to_index.get(away)
        if home_idx is not None and away_idx is not None:
            remaining_matches_info.append({'home_idx': home_idx, 'away_idx': away_idx,
                                           'home_team': home, 'away_team': away,
                                           'match_num_orig': row['Match Number']})
        else:
            print(f"Warning: Skipping simulation for match {row.get('Match Number', 'N/A')} due to unknown team(s): {home} vs {away}")

    num_remaining_matches_sim = len(remaining_matches_info)
    if num_remaining_matches != num_remaining_matches_sim:
         print(f"Note: Simulating {num_remaining_matches_sim} matches ({num_remaining_matches - num_remaining_matches_sim} skipped due to unknown teams).")

    if num_remaining_matches_sim == 0:
        print("No valid remaining matches to simulate after filtering. Final standings are above.")
        sys.exit()

except Exception as e:
    print(f"FATAL ERROR: Failed preparing remaining matches: {e}")
    traceback.print_exc()
    sys.exit("Exiting: Could not prepare simulation data.")

# --- Brute-Force Simulation ---
print(f"\n--- Starting Brute-Force Simulation for {num_remaining_matches_sim} matches ---")

# Safety check for number of scenarios
max_matches_for_brute_force = 22 # Set a practical limit
num_scenarios = 2**num_remaining_matches_sim

if num_remaining_matches_sim > max_matches_for_brute_force:
    print(f"FATAL ERROR: Too many remaining matches ({num_remaining_matches_sim}) for brute-force simulation.")
    print(f"This would generate {num_scenarios:,.0f} scenarios, which is computationally infeasible.")
    print(f"Consider using the Monte Carlo simulation for >{max_matches_for_brute_force} remaining matches.")
    sys.exit(f"Exiting: Brute-force scenario count too high.")

print(f"Total Scenarios to evaluate: {num_scenarios:,.0f}")
sim_start_time = time.time()

# Initialize qualification counts
top_4_qualifications = {team: 0 for team in teams}
exact_rank_counts = {team: [0] * num_teams for team in teams} # team: [rank1_count, rank2_count, ...]
points_distribution = {team: {} for team in teams} # team: {points: count}

# Iterate through all possible outcomes of remaining matches
# Each 'scenario' is a tuple of winners (0 for home, 1 for away) for each remaining match
scenario_num = 0
for scenario_outcomes in product([0, 1], repeat=num_remaining_matches_sim):
    scenario_num +=1
    if scenario_num % 100000 == 0: # Progress update
        elapsed_time = time.time() - sim_start_time
        scenarios_per_sec = scenario_num / elapsed_time if elapsed_time > 0 else 0
        print(f"  ... Evaluated {scenario_num:,.0f}/{num_scenarios:,.0f} scenarios ({scenarios_per_sec:.0f} scenarios/sec). "
              f"Elapsed: {elapsed_time:.2f}s", end='\r')

    current_points = np.copy(initial_points_np)

    # Update points table for this specific scenario
    for i, outcome in enumerate(scenario_outcomes):
        match_info = remaining_matches_info[i]
        home_idx = match_info['home_idx']
        away_idx = match_info['away_idx']

        current_points[home_idx, 0] += 1 # Matches played
        current_points[away_idx, 0] += 1 # Matches played

        if outcome == 0: # Home team wins
            current_points[home_idx, 1] += 1 # Wins
            current_points[home_idx, 3] += 2 # Points
            current_points[away_idx, 2] += 1 # Losses
        else: # Away team wins
            current_points[away_idx, 1] += 1 # Wins
            current_points[away_idx, 3] += 2 # Points
            current_points[home_idx, 2] += 1 # Losses

    # Create a DataFrame for sorting
    scenario_df = pd.DataFrame(
        current_points,
        columns=['M', 'W', 'L', 'Pts'],
        index=[index_to_team[j] for j in range(num_teams)]
    )
    # Tie-breaking: Points, then Wins. NRR is not considered here.
    scenario_df = scenario_df.sort_values(by=['Pts', 'W'], ascending=[False, False])

    # Record points distribution
    for team_idx in range(num_teams):
        team_name = index_to_team[team_idx]
        team_points = current_points[team_idx, 3]
        points_distribution[team_name][team_points] = points_distribution[team_name].get(team_points, 0) + 1

    # Record top 4 qualifications
    top_4_teams_scenario = scenario_df.index[:4].tolist()
    for team in top_4_teams_scenario:
        top_4_qualifications[team] += 1

    # Record exact rank
    for rank, team_name in enumerate(scenario_df.index):
        exact_rank_counts[team_name][rank] += 1

sim_end_time = time.time()
print(f"\nBrute-Force Simulation Completed. Total scenarios: {num_scenarios:,.0f}")
print(f"Total simulation time: {sim_end_time - sim_start_time:.2f} seconds.")
print("-" * 50)

# --- Display Results ---
print("Top 4 Qualification Probabilities (Brute-Force):")
results_data = []
for team in teams:
    count = top_4_qualifications.get(team, 0)
    probability = (count / num_scenarios) * 100 if num_scenarios > 0 else 0
    results_data.append({
        'Team': team,
        'Top 4 Finishes': f"{int(count):,}",
        'Probability (%)': probability
    })

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values(by='Probability (%)', ascending=False)
results_df_display = results_df.copy()
results_df_display['Probability (%)'] = results_df_display['Probability (%)'].map('{:.2f}%'.format)
print(results_df_display.to_string(index=False))
print("-" * 50)

# --- Detailed Rank Probabilities ---
print("\nExact Rank Probabilities:")
rank_prob_data = []
for team in teams:
    row = {'Team': team}
    for rank_idx in range(num_teams):
        count = exact_rank_counts[team][rank_idx]
        prob = (count / num_scenarios) * 100 if num_scenarios > 0 else 0
        row[f'Rank {rank_idx+1} (%)'] = f"{prob:.2f}"
    rank_prob_data.append(row)

rank_prob_df = pd.DataFrame(rank_prob_data)
# Reorder columns to Team, Rank 1, Rank 2, ...
cols = ['Team'] + [f'Rank {i+1} (%)' for i in range(num_teams)]
rank_prob_df = rank_prob_df[cols]
print(rank_prob_df.to_string(index=False))
print("-" * 50)


# --- Points Distribution Visualization (Optional) ---
# This can be very verbose if plotted for all teams.
# Consider plotting for a few key teams or if num_teams is small.
plot_points_dist = True # Set to False to disable plotting
if plot_points_dist:
    print("\nPoints Distribution Plots:")
    num_plot_cols = 3
    num_plot_rows = (num_teams + num_plot_cols - 1) // num_plot_cols
    fig, axes = plt.subplots(num_plot_rows, num_plot_cols, figsize=(5 * num_plot_cols, 4 * num_plot_rows))
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    for i, team in enumerate(teams):
        ax = axes[i]
        dist = points_distribution[team]
        if not dist:
            ax.text(0.5, 0.5, 'No scenarios', ha='center', va='center')
            ax.set_title(team)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        points = sorted(dist.keys())
        counts = [dist[p] for p in points]
        probabilities = [c / num_scenarios * 100 for c in counts]

        sns.barplot(x=points, y=probabilities, ax=ax, color='skyblue')
        ax.set_title(f"{team} - Points Distribution")
        ax.set_xlabel("Points")
        ax.set_ylabel("Probability (%)")
        ax.tick_params(axis='x', rotation=45)
        for p_bar, prob_val in zip(ax.patches, probabilities):
            ax.text(p_bar.get_x() + p_bar.get_width() / 2.,
                    p_bar.get_height(),
                    f'{prob_val:.1f}%',
                    ha='center', va='bottom', fontsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # Save the plot to a file in the /kaggle/working/ directory
    plot_filename = "/kaggle/working/points_distribution.png"
    try:
        plt.savefig(plot_filename)
        print(f"\nPoints distribution plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nError saving points distribution plot: {e}")
    # plt.show() # Typically not used in scripts, but useful in notebooks

print("\nNotes:")
print(" - Probabilities are exact based on all possible outcomes (brute-force).")
print(" - Assumes a 50/50 chance for each team in every remaining match.")
print(" - Ranking uses Points, then Wins. NRR is not considered.")
print("=" * 50)
print("Script finished.")

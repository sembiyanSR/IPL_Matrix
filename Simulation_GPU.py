# Required libraries
import numpy as np
import pandas as pd
import time
import sys
import os # For checking file paths if needed
import traceback # For printing full tracebacks on error

# --- GPU Check and Initialization ---
try:
    import cupy as cp
    print(f"Successfully imported CuPy version: {cp.__version__}")
except ImportError:
    print("FATAL ERROR: CuPy library not found.")
    print("Please ensure CuPy is installed in your Kaggle environment.")
    print("Go to 'Settings' -> 'Accelerator' and select a GPU (e.g., P100).")
    sys.exit("Exiting: CuPy required for GPU execution.")
except Exception as e:
    print(f"FATAL ERROR: An unexpected error occurred during CuPy import: {e}")
    sys.exit("Exiting: CuPy import failed.")

if not cp.cuda.is_available():
    print("FATAL ERROR: CuPy is imported, but no CUDA-enabled GPU is detected.")
    print("Please ensure you have selected a GPU Accelerator in Kaggle Settings.")
    sys.exit("Exiting: No GPU available for CuPy.")
else:
    try:
        device_id = 0
        cp.cuda.Device(device_id).use()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        gpu_name = props['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        print(f"GPU Detected and Initialized: ID {device_id} - {gpu_name}")
        try:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            print(f"GPU Memory: {free_mem / 1024**3:.2f} GB Free / {total_mem / 1024**3:.2f} GB Total")
        except Exception as mem_e:
            print(f"Warning: Could not get GPU memory info: {mem_e}")
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize GPU device {device_id}.")
        print(f"Error details: {e}")
        sys.exit("Exiting: GPU initialization failed.")

# --- Code Execution Starts Here ---
print("\nGPU checks passed. Proceeding with data loading and simulation...")

# --- Data Loading and Preprocessing ---
try:
    file_path = '/kaggle/input/ipl2025fixtures/ipl-2025-UTC.csv'
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"Schedule file not found at: {file_path}")

    df_orig = pd.read_csv(file_path)
    df_orig = df_orig.loc[pd.to_numeric(df_orig['Match Number'], errors='coerce').notna()].copy()
    df_orig['Match Number'] = df_orig['Match Number'].astype(int)
    df_orig = df_orig.drop(columns=['Round Number', 'Date', 'Location'], errors='ignore')
    df_orig = df_orig[:-4]
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
    sys.exit("Exiting: Data processing failed.")

# --- Function to Update Results ---
def update_result(df, match_number, winner):
    winner_abbr = team_abbreviations.get(winner, winner)
    if match_number in df['Match Number'].values:
        df.loc[df['Match Number'] == match_number, 'Result'] = winner_abbr
    else:
        print(f"Warning: Match Number {match_number} not found in schedule.")
    return df

# ==============================================================================
# == UPDATE COMPLETED MATCH RESULTS HERE ==
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
# ... add more results as they happen
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

    completed_matches = df_abbreviated[df_abbreviated['Result'] != ''].copy()
    initial_points_np = np.zeros((num_teams, 4), dtype=np.int32)

    for _, row in completed_matches.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        winner = row['Result']
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

except Exception as e:
    print(f"FATAL ERROR: Failed during initial standings calculation: {e}")
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
    num_total_remaining = len(remaining_matches_df)
    remaining_matches_indices_list = []
    for _, row in remaining_matches_df.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        home_idx = team_to_index.get(home)
        away_idx = team_to_index.get(away)
        if home_idx is not None and away_idx is not None:
            remaining_matches_indices_list.append([home_idx, away_idx])
        else:
            print(f"Warning: Skipping simulation for match {row.get('Match Number', 'N/A')} due to unknown team(s): {home} vs {away}")

    remaining_matches_indices_np = np.array(remaining_matches_indices_list, dtype=np.int32)
    num_remaining_matches_sim = len(remaining_matches_indices_np)

    if num_total_remaining != num_remaining_matches_sim:
         print(f"Note: Simulating {num_remaining_matches_sim} matches ({num_total_remaining - num_remaining_matches_sim} skipped due to unknown teams).")

except Exception as e:
    print(f"FATAL ERROR: Failed preparing remaining matches: {e}")
    sys.exit("Exiting: Could not prepare simulation data.")

# --- Monte Carlo Simulation ---
if num_remaining_matches_sim == 0:
    print("No remaining valid matches to simulate. Final standings are above.")
else:
    # --- Simulation Settings ---
    num_simulations = 5_000_000_000
    gpu_batch_size = 1_000_000

    print("--- Starting GPU Monte Carlo Simulation ---")
    num_batches = (num_simulations + gpu_batch_size - 1) // gpu_batch_size
    print(f"Total Simulations: {num_simulations:,}")
    print(f"GPU Batch Size: {gpu_batch_size:,}")
    print(f"Number of Batches: {num_batches}")
    print(f"Simulating {num_remaining_matches_sim} remaining matches per scenario.")

    sim_start_time = time.time()

    # Transfer fixed data to GPU only once
    try:
        cp_initial_points = cp.asarray(initial_points_np)
        cp_remaining_matches_indices = cp.asarray(remaining_matches_indices_np)
    except Exception as e:
        print(f"FATAL ERROR: Failed to transfer initial data to GPU: {e}")
        sys.exit("Exiting: GPU data transfer failed.")

    cp_top_4_qualifications = cp.zeros(num_teams, dtype=cp.int64)

    # --- Main GPU Simulation Loop (Batched) ---
    current_batch_num = 0 # For error reporting
    try:
        for i_batch in range(num_batches):
            current_batch_num = i_batch + 1
            batch_start_time = time.time()
            current_batch_size = min(gpu_batch_size, num_simulations - i_batch * gpu_batch_size)
            if current_batch_size <= 0: continue

            # 1. Simulate outcomes
            cp_match_outcomes = cp.random.randint(0, 2,
                                                 size=(current_batch_size, num_remaining_matches_sim),
                                                 dtype=cp.int8)

            # 2. Initialize points tables
            cp_batch_points = cp.repeat(cp_initial_points[cp.newaxis, :, :], current_batch_size, axis=0)

            # 3. Apply results
            batch_indices_gpu = cp.arange(current_batch_size)
            for match_idx in range(num_remaining_matches_sim):
                home_team_idx = cp_remaining_matches_indices[match_idx, 0]
                away_team_idx = cp_remaining_matches_indices[match_idx, 1]
                home_wins_mask = (cp_match_outcomes[:, match_idx] == 1)
                away_wins_mask = ~home_wins_mask
                # Update Matches
                cp_batch_points[batch_indices_gpu, home_team_idx, 0] += 1
                cp_batch_points[batch_indices_gpu, away_team_idx, 0] += 1
                # Update Wins/Points/Losses
                cp_batch_points[home_wins_mask, home_team_idx, 1] += 1
                cp_batch_points[home_wins_mask, home_team_idx, 3] += 2
                cp_batch_points[home_wins_mask, away_team_idx, 2] += 1
                cp_batch_points[away_wins_mask, away_team_idx, 1] += 1
                cp_batch_points[away_wins_mask, away_team_idx, 3] += 2
                cp_batch_points[away_wins_mask, home_team_idx, 2] += 1

            # 4. Determine Top 4 using GPU argsort with a composite key
            points_col = cp_batch_points[:, :, 3] # Shape: (batch_size, num_teams)
            wins_col = cp_batch_points[:, :, 1]   # Shape: (batch_size, num_teams)

            # Define a multiplier > max possible wins (e.g., 100 is safe)
            multiplier = 100
            # Create composite key: Points dominate, Wins break ties
            composite_key = (points_col * multiplier) + wins_col # Shape: (batch_size, num_teams)

            # Sort descending based on the composite key along the teams axis (axis=1)
            # Use negative key for descending sort with argsort
            sorted_indices = cp.argsort(-composite_key, axis=1)
            # sorted_indices shape: (batch_size, num_teams)

            # Get the indices of the top 4 teams for each simulation
            top_4_indices_batch = sorted_indices[:, :4] # Shape: (batch_size, 4)

            # 5. Accumulate Top 4 counts
            counts = cp.bincount(top_4_indices_batch.ravel(), minlength=num_teams)
            cp_top_4_qualifications += counts.astype(cp_top_4_qualifications.dtype)

            # --- Progress Indicator & Memory Cleanup ---
            batch_end_time = time.time()
            sims_done = min((i_batch + 1) * gpu_batch_size, num_simulations)
            print(f"  ... Batch {i_batch+1}/{num_batches} ({current_batch_size:,} sims) completed in "
                  f"{batch_end_time - batch_start_time:.2f}s. Total Sims: {sims_done:,}/{num_simulations:,}")

            # Explicitly delete intermediate arrays from this batch
            del cp_batch_points, cp_match_outcomes, points_col, wins_col, multiplier, composite_key, sorted_indices, top_4_indices_batch, counts, home_wins_mask, away_wins_mask, batch_indices_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    except Exception as e:
        print(f"\nFATAL ERROR: An error occurred during GPU simulation loop (Batch {current_batch_num}): {e}")
        traceback.print_exc() # Print the detailed traceback
        try: # Attempt cleanup
            if 'cp_initial_points' in locals(): del cp_initial_points
            if 'cp_remaining_matches_indices' in locals(): del cp_remaining_matches_indices
            if 'cp_top_4_qualifications' in locals(): del cp_top_4_qualifications
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as cleanup_e:
            print(f"Additionally, an error occurred during cleanup: {cleanup_e}")
        sys.exit("Exiting: Simulation failed.")

    # --- Simulation Complete - Process Results ---
    sim_end_time = time.time()
    print("-" * 50)
    print(f"GPU Monte Carlo Simulation Completed.")
    print(f"Total simulation time: {sim_end_time - sim_start_time:.2f} seconds for {num_simulations:,} scenarios.")
    print("-" * 50)

    # Transfer final counts back to CPU
    try:
        top_4_qualifications_np = cp.asnumpy(cp_top_4_qualifications)
    except Exception as e:
        print(f"FATAL ERROR: Failed to transfer results from GPU to CPU: {e}")
        sys.exit("Exiting: Result transfer failed.")

    # Convert final counts to dictionary for display
    top_4_qualifications = {index_to_team.get(i, f"Unknown_{i}"): count
                            for i, count in enumerate(top_4_qualifications_np)}

    # --- Display Results ---
    print("Estimated Top 4 Qualification Probabilities:")
    results_data = []
    for team in teams:
        count = top_4_qualifications.get(team, 0)
        probability = (count / num_simulations) * 100 if num_simulations > 0 else 0
        results_data.append({
            'Team': team,
            'Top 4 Finishes (Est)': f"{int(count):,}",
            'Probability (%)': probability
        })

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(by='Probability (%)', ascending=False)
    results_df_display = results_df.copy()
    results_df_display['Probability (%)'] = results_df_display['Probability (%)'].map('{:.2f}%'.format)
    print(results_df_display.to_string(index=False))

    print("\nNotes:")
    print(" - Probabilities are estimates based on Monte Carlo simulation.")
    print(" - Assumes a 50/50 chance for each team in every remaining match.")
    print(" - Ranking uses Points, ignores actual NRR ")
    print(" - Simulation performed exclusively on GPU using CuPy.")
    print("=" * 50)

# --- Final Cleanup ---
try:
    if 'cp_initial_points' in locals(): del cp_initial_points
    if 'cp_remaining_matches_indices' in locals(): del cp_remaining_matches_indices
    if 'cp_top_4_qualifications' in locals(): del cp_top_4_qualifications
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print("Final GPU memory resources released.")
except NameError:
    print("Final GPU memory cleanup skipped (variables might not be defined).")
except Exception as e:
    print(f"An error occurred during final GPU memory cleanup: {e}")

print("Script finished.")

 

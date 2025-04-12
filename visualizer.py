import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_convergence_log(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def plot_error_and_residuals_over_time(convergence_log):
    # X-axis: epoch index (time)
    epoch_indices = []
    errors = []
    residuals = []  # List of (epoch_idx, sv_label, value)

    for epoch_idx, epoch_data in tqdm(list(enumerate(convergence_log)), desc="Processing epochs"):
        convergence = epoch_data["convergence"]
        if not convergence:
            continue
        # Use the last iteration's error for this epoch
        last = convergence[-1]
        error = last.get("error_from_rinex_header_m", None)
        errors.append(error)
        epoch_indices.append(epoch_idx)
        # Handle both dict and list for pseudorange residuals
        omc = last.get("omc", {})
        if isinstance(omc, dict):
            for sv, value in omc.items():
                residuals.append((epoch_idx, sv, value))
        elif isinstance(omc, list):
            # Try to get SV labels if present
            sv_labels = epoch_data.get("sv_labels", [f"SV {i+1}" for i in range(len(omc))])
            for i, value in enumerate(omc):
                residuals.append((epoch_idx, sv_labels[i], value))

    # Prepare data for plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot error over time
    ax1.plot(epoch_indices, errors, marker='o', color='tab:red')
    ax1.set_ylabel("Position Error (meters)")
    ax1.set_title("Position Error Over Time (Epoch Index)")
    min_idx = int(np.nanargmin([e if e is not None else np.nan for e in errors]))
    ax1.annotate("Min Error", (epoch_indices[min_idx], errors[min_idx]), textcoords="offset points", xytext=(0,10), ha='center', color='blue', fontsize=9)

    # Plot pseudorange residuals over time (scatter, colored by SV)
    sv_set = sorted(set(sv for _, sv, _ in residuals))
    colors = plt.cm.get_cmap('tab20', len(sv_set))
    for i, sv in enumerate(tqdm(sv_set, desc="Plotting satellites", leave=False)):
        x = [epoch_idx for epoch_idx, sv_label, _ in residuals if sv_label == sv]
        y = [value for epoch_idx, sv_label, value in residuals if sv_label == sv]
        ax2.scatter(x, y, label=str(sv), color=colors(i), s=20)
    ax2.set_ylabel("Pseudorange Residual (meters)")
    ax2.set_xlabel("Epoch Index")
    ax2.set_title("Pseudorange Residuals Over Time (All Satellites)")
    ax2.legend(title="SV", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()

def main():
    json_path = "spp_convergence_log.json"
    if not os.path.exists(json_path):
        print(f"Convergence log file '{json_path}' not found.")
        return

    convergence_log = load_convergence_log(json_path)
    print(f"Loaded {len(convergence_log)} epochs from {json_path}.")

    plot_error_and_residuals_over_time(convergence_log)

if __name__ == "__main__":
    main()
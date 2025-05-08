import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_spp_results(json_path):
    """Loads SPP results from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

def plot_final_error_and_residuals(spp_results):
    """Plots final position error and pseudorange residuals from SPP results."""
    epoch_indices = []
    errors = []
    residuals = []  # List of (epoch_idx, sv_label, value)

    # data from spp_results
    for epoch_idx, epoch_data in enumerate(spp_results):
        error = epoch_data.get("error_from_rinex_header_m", None)
        errors.append(error)
        epoch_indices.append(epoch_idx)
        
        residuals_m = epoch_data.get("residuals_m", {})
        for sv_label, value in residuals_m.items():
            residuals.append((epoch_idx, sv_label, value))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    valid_errors = [(idx, err) for idx, err in zip(epoch_indices, errors) if err is not None]
    if valid_errors:
        valid_indices, valid_error_values = zip(*valid_errors)
        ax1.plot(valid_indices, valid_error_values, color='tab:red', linestyle='-')
        min_idx_local = np.argmin(valid_error_values)
        min_epoch_idx = valid_indices[min_idx_local]
        min_error_val = valid_error_values[min_idx_local]
        ax1.annotate(f"Min Error: {min_error_val:.2f}m", (min_epoch_idx, min_error_val), 
                     textcoords="offset points", xytext=(0,10), ha='center', color='blue', fontsize=9)
    ax1.set_ylabel("Position Error from RINEX Header (m)")
    ax1.set_title("Final Position Error Over Time (Epoch Index)")
    ax1.grid(True)

    # Plot pseudorange residuals
    if residuals:
        sv_set = sorted(set(sv for _, sv, _ in residuals))
        
        colors = plt.cm.get_cmap('turbo', len(sv_set)) if len(sv_set) > 20 else plt.cm.get_cmap('tab20', len(sv_set))
        for i, sv in enumerate(sv_set):
            x = [epoch_idx for epoch_idx, sv_label, _ in residuals if sv_label == sv]
            y = [value for epoch_idx, sv_label, value in residuals if sv_label == sv]
            ax2.scatter(x, y, label=str(sv), color=colors(i), s=15, alpha=0.7)
        ax2.legend(title="SV", bbox_to_anchor=(1, 1))
    ax2.set_ylabel("Pseudorange Residual (m)")
    ax2.set_xlabel("Epoch Index")
    ax2.set_title("Pseudorange Residuals Over Time (All Satellites)")
    ax2.grid(True)

    plt.show()

def main():
    json_path = "spp_results.json"
    if not os.path.exists(json_path):
        print(f"SPP results file '{json_path}' not found.")
        return

    spp_results = load_spp_results(json_path)
    if not spp_results:
        print(f"No data found in {json_path}.")
        return
        
    print(f"Loaded {len(spp_results)} epochs from {json_path}.")

    plot_final_error_and_residuals(spp_results)

if __name__ == "__main__":
    main()
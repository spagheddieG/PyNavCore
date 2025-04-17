import georinex as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Load RINEX File ----
filename = "algo0010.25o"  # Replace with your actual file path
obs = gr.load(filename)

# ---- Select Observables ----
# First print available variables for debugging
print("Available variables:", list(obs.data_vars))

# Access observables directly as variables
snr = obs['S1']  # Signal strength (SNR for L1)
phase = obs['L1']  # Carrier phase L1
pseudo = obs['C1']  # Pseudorange L1

# ---- Preprocess ----
satellites = snr.sv.values
time = snr.time.values

# ---- Detect SNR Drops ----
def detect_snr_drops(snr_data, threshold=30):
    drops = (snr_data < threshold)
    return drops.sum(dim="sv")  # total satellites below threshold per time

# ---- Detect Carrier Phase Slips ----
def detect_phase_slips(phase_data, slip_threshold=0.05):
    diff = phase_data.diff(dim="time")
    slips = (np.abs(diff) > slip_threshold)
    return slips.sum(dim="sv")

# ---- Detect Pseudorange Jumps ----
def detect_pseudorange_jumps(pseudo_data, jump_threshold=10):
    diff = pseudo_data.diff(dim="time")
    jumps = (np.abs(diff) > jump_threshold)
    return jumps.sum(dim="sv")

# ---- Run Detection ----
snr_drops = detect_snr_drops(snr)
phase_slips = detect_phase_slips(phase)
pseudo_jumps = detect_pseudorange_jumps(pseudo)

# ---- Combine Metrics ----
jam_score = snr_drops + phase_slips + pseudo_jumps
jam_detected = jam_score > 3  # Adjust threshold as needed

# ---- Plot Results ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# Plot 1: Jamming Score
ax1.plot(obs.time[1:], jam_score, label="Jamming Score")
ax1.axhline(3, color='red', linestyle='--', label="Threshold")
ax1.set_title("GNSS Jamming Detection")
ax1.set_ylabel("Combined Score")
ax1.legend()
ax1.grid()

# Plot 2: L1 vs L2 Phase
l2_phase = obs['L2']  # Get L2 carrier phase data
ax2.plot(obs.time, phase.mean(dim='sv'), label="L1 Phase")
ax2.plot(obs.time, l2_phase.mean(dim='sv'), label="L2 Phase")
ax2.set_title("L1 vs L2 Carrier Phase Observations")
ax2.set_ylabel("Phase (cycles)")
ax2.legend()
ax2.grid()

# Plot 3: L1 vs L2 Pseudorange
l2_pseudo = obs['C2']  # Get L2 pseudorange data
ax3.plot(obs.time, pseudo.mean(dim='sv'), label="L1 Pseudorange (C1)")
ax3.plot(obs.time, l2_pseudo.mean(dim='sv'), label="L2 Pseudorange (C2)")
ax3.set_title("L1 vs L2 Pseudorange Observations")
ax3.set_xlabel("Time")
ax3.set_ylabel("Pseudorange (m)")
ax3.legend()
ax3.grid()
plt.tight_layout()
plt.show()

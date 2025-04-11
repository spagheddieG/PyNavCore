"""
spp_module.py

This module provides a modular Single Point Positioning (SPP) solver for GNSS (Global Navigation Satellite System) data.
It includes functions to:
- Calculate satellite positions and clock corrections from broadcast ephemeris.
- Parse receiver approximate position from RINEX observation file headers.
- Solve for receiver position and clock offset using pseudorange measurements and navigation data.

Dependencies:
- georinex: For reading RINEX observation and navigation files.
- numpy, xarray: For numerical and array operations.
- Standard Python libraries: datetime, math, logging, json, typing.

Author: Eddie G
"""

import georinex as gr
import numpy as np
import xarray as xr
from datetime import datetime
import math
import logging
import json
from typing import Optional, List, Dict, Any

# --- Constants ---
C = 299792458.0  # Speed of light in vacuum (meters per second)
GM = 3.986005e14  # WGS84 gravitational constant (m^3/s^2)
OMEGA_E_DOT = 7.2921151467e-5  # WGS84 Earth rotation rate (radians per second)

def calculate_satellite_position_and_clock(ephem, transmit_time_gps_week, transmit_time_sow):
    """
    Compute the ECEF position and clock correction for a GPS satellite at a given transmit time.

    Parameters:
        ephem: xarray Dataset or dict-like object containing broadcast ephemeris for the satellite.
        transmit_time_gps_week: GPS week number (not used in this function, but may be for future extension).
        transmit_time_sow: GPS seconds of week at signal transmission.

    Returns:
        sat_pos_ecef: np.ndarray of shape (3,), satellite ECEF position in meters.
        sat_clock_corr: float, satellite clock correction in seconds.
        Returns (None, None) if computation fails or does not converge.
    """
    try:
        # Time of ephemeris (Toe) and clock (Toc)
        toe = ephem['Toe'].item()
        try:
            toc = ephem['Toc'].item()
        except KeyError:
            toc = ephem['Toe'].item()
        # Time from ephemeris reference epoch
        tk = transmit_time_sow - toe
        # Account for beginning/end of GPS week crossover
        if tk > 302400:
            tk -= 604800
        elif tk < -302400:
            tk += 604800

        # Satellite clock correction parameters
        af0 = ephem['SVclockBias'].item()
        af1 = ephem['SVclockDrift'].item()
        af2 = ephem['SVclockDriftRate'].item()
        dt_clock = transmit_time_sow - toc
        if dt_clock > 302400:
            dt_clock -= 604800
        elif dt_clock < -302400:
            dt_clock += 604800
        # Satellite clock bias (seconds)
        sat_clock_bias = af0 + af1 * dt_clock + af2 * dt_clock**2

        # Semi-major axis
        a = ephem['sqrtA'].item()**2
        # Computed mean motion (rad/s)
        n0 = math.sqrt(GM / a**3)
        # Corrected mean motion
        n = n0 + ephem['DeltaN'].item()
        # Mean anomaly at tk
        Mk = ephem['M0'].item() + n * tk

        # Solve Kepler's equation for eccentric anomaly (Ek) using iterative method
        Ek = Mk
        for _ in range(10):
            Ek_old = Ek
            Ek = Mk + ephem['Eccentricity'].item() * math.sin(Ek_old)
            if abs(Ek - Ek_old) < 1e-12:
                break
        else:
            # If not converged, return None
            return None, None

        sin_Ek = math.sin(Ek)
        cos_Ek = math.cos(Ek)
        e = ephem['Eccentricity'].item()
        # True anomaly (vk)
        vk_num = math.sqrt(1 - e**2) * sin_Ek
        vk_den = cos_Ek - e
        vk = math.atan2(vk_num, vk_den)
        # Argument of latitude
        Phik = vk + ephem['omega'].item()
        sin2Phik = math.sin(2 * Phik)
        cos2Phik = math.cos(2 * Phik)
        # Corrections for argument of latitude, radius, and inclination
        duk = ephem['Cuc'].item() * cos2Phik + ephem['Cus'].item() * sin2Phik
        drk = ephem['Crc'].item() * cos2Phik + ephem['Crs'].item() * sin2Phik
        dik = ephem['Cic'].item() * cos2Phik + ephem['Cis'].item() * sin2Phik
        # Corrected argument of latitude, radius, and inclination
        uk = Phik + duk
        rk = a * (1 - e * cos_Ek) + drk
        ik = ephem['Io'].item() + ephem['IDOT'].item() * tk + dik
        # Positions in orbital plane
        xk_prime = rk * math.cos(uk)
        yk_prime = rk * math.sin(uk)
        # Corrected longitude of ascending node
        Omega_k = ephem['Omega0'].item() + (ephem['OmegaDot'].item() - OMEGA_E_DOT) * tk - OMEGA_E_DOT * toe
        cos_Omega_k = math.cos(Omega_k)
        sin_Omega_k = math.sin(Omega_k)
        cos_ik = math.cos(ik)
        sin_ik = math.sin(ik)
        # ECEF coordinates
        Xk = xk_prime * cos_Omega_k - yk_prime * cos_ik * sin_Omega_k
        Yk = xk_prime * sin_Omega_k + yk_prime * cos_ik * cos_Omega_k
        Zk = yk_prime * sin_ik
        sat_pos_ecef = np.array([Xk, Yk, Zk])

        # Relativistic correction (meters to seconds)
        relativistic_corr = -2 * math.sqrt(GM * a) * e * sin_Ek / C**2
        # Total satellite clock correction (seconds)
        sat_clock_corr = sat_clock_bias + relativistic_corr - ephem['TGD'].item()
        return sat_pos_ecef, sat_clock_corr
    except Exception:
        # If any error occurs, return None
        return None, None

def parse_rinex_header_xyz(obs_file: str) -> Optional[List[float]]:
    """
    Parse the approximate receiver position (XYZ) from the header of a RINEX observation file.

    Parameters:
        obs_file: Path to the RINEX observation file.

    Returns:
        List of [X, Y, Z] ECEF coordinates (meters) if found, else None.
    """
    try:
        with open(obs_file, 'r') as f:
            for line in f:
                if "APPROX POSITION XYZ" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        pass
    return None

def spp_solve(
    obs_file: str,
    nav_file: str,
    pseudorange_code: str = 'C1',
    max_epochs: Optional[int] = None,
    min_sats: int = 4,
    convergence_threshold: float = 1e-4,
    max_iterations: int = 10,
    initial_xyz: Optional[List[float]] = None,
    output_json: str = "spp_results.json"
) -> List[Dict[str, Any]]:
    """
    Modular SPP (Single Point Positioning) solver.

    Processes GNSS RINEX observation and navigation files to estimate receiver position and clock offset
    for each epoch, using broadcast ephemeris and pseudorange measurements.

    Results are written to a JSON file, including error from RINEX header position, epoch time, estimated
    position, clock offset, and number of satellites used.

    Parameters:
        obs_file: Path to RINEX observation file.
        nav_file: Path to RINEX navigation file.
        pseudorange_code: Observation code for pseudorange (default 'C1').
        max_epochs: Maximum number of epochs to process (None for all).
        min_sats: Minimum number of satellites required for a solution (default 4).
        convergence_threshold: Position convergence threshold in meters (default 1e-4).
        max_iterations: Maximum number of iterations for least-squares solver (default 10).
        initial_xyz: Optional initial receiver position [X, Y, Z] in meters.
        output_json: Output JSON file path for results.

    Returns:
        List of dictionaries, one per processed epoch, with keys:
            - "epoch": ISO8601 string of epoch time
            - "position_ecef": [X, Y, Z] in meters
            - "receiver_clock_offset_ns": receiver clock offset in nanoseconds
            - "num_sats": number of satellites used
            - "error_from_rinex_header_m": error from RINEX header position (meters), if available
    """
    # Set up logging for info messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load RINEX observation and navigation files (GPS only for obs)
    obs = gr.load(obs_file, use='G')
    nav = gr.load(nav_file)

    # Get receiver's approximate position from RINEX header, if available
    rinex_xyz = None
    try:
        rinex_xyz = obs.attrs['position_xyz']
        if isinstance(rinex_xyz, np.ndarray):
            rinex_xyz = rinex_xyz.tolist()
        elif isinstance(rinex_xyz, (xr.DataArray, xr.Variable)):
            rinex_xyz = rinex_xyz.values.tolist()
    except Exception:
        rinex_xyz = None
    if rinex_xyz is None:
        rinex_xyz = parse_rinex_header_xyz(obs_file)

    # Use Dallas, TX as default initial position if not provided
    if initial_xyz is None:
        initial_xyz = [-1288392.5, -4865182.1, 3999769.7]

    results = []
    processed_epochs = 0
    # Get unique epoch times from observation file
    unique_times = np.unique(obs['time'].values)

    for epoch_time_dt64 in unique_times:
        # Limit number of processed epochs if max_epochs is set
        if max_epochs is not None and processed_epochs >= max_epochs:
            break

        # Convert epoch time to datetime
        epoch_time = gr.to_datetime(epoch_time_dt64)
        epoch_obs = obs.sel(time=epoch_time_dt64)

        # Identify satellites with valid pseudorange for this epoch
        valid_svs = epoch_obs['sv'][np.isfinite(epoch_obs[pseudorange_code])].values
        if len(valid_svs) < min_sats:
            continue  # Not enough satellites for a solution

        # Select only valid satellites
        epoch_obs = epoch_obs.sel(sv=valid_svs)
        sat_positions = []
        sat_clock_corrections = []
        pseudoranges = []

        # Compute GPS week and seconds of week for this epoch
        GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
        if isinstance(epoch_time, np.datetime64):
            epoch_time = gr.to_datetime(epoch_time)
        if not isinstance(epoch_time, datetime):
            try:
                import pandas as pd
                epoch_time = pd.to_datetime(epoch_time).to_pydatetime()
            except Exception:
                continue
        if hasattr(epoch_time, "tzinfo") and epoch_time.tzinfo is not None:
            epoch_time = epoch_time.replace(tzinfo=None)
        time_diff = epoch_time - GPS_EPOCH
        total_seconds = time_diff.total_seconds()
        gps_week = int(total_seconds // (7 * 24 * 3600))
        time_of_week = total_seconds % (7 * 24 * 3600)

        # For each satellite, compute position and clock correction
        for sv in epoch_obs['sv'].values:
            pr = epoch_obs[pseudorange_code].sel(sv=sv).item()
            # Select latest available ephemeris for this satellite before or at this epoch
            available_ephem = nav.sel(sv=sv, time=slice(None, epoch_time_dt64)).dropna(dim='time', how='all')
            if available_ephem.time.size == 0:
                continue  # No ephemeris available
            ephem = available_ephem.isel(time=-1)
            # Approximate distance from initial position (not used in calculation, but could be for selection)
            approx_dist = np.linalg.norm(initial_xyz)
            # Estimate transmit time (seconds of week) by correcting for signal travel time
            transmit_time_sow = time_of_week - pr / C
            # Compute satellite ECEF position and clock correction
            sat_pos, sat_clk = calculate_satellite_position_and_clock(ephem, gps_week, transmit_time_sow)
            if sat_pos is not None and sat_clk is not None:
                # Correct for Earth's rotation during signal travel (Sagnac effect)
                omega_tau = OMEGA_E_DOT * (pr / C)
                Rz = np.array([[ math.cos(omega_tau), math.sin(omega_tau), 0],
                               [-math.sin(omega_tau), math.cos(omega_tau), 0],
                               [ 0,                0,               1]])
                sat_pos_corrected = Rz @ sat_pos
                sat_positions.append(sat_pos_corrected)
                sat_clock_corrections.append(sat_clk)
                pseudoranges.append(pr)

        if len(sat_positions) < min_sats:
            continue  # Not enough satellites for a solution

        num_sats = len(sat_positions)
        # Initialize state vector: [X, Y, Z, clock bias]
        x = np.zeros(4)
        current_pos = np.array(initial_xyz)
        delta_x = None

        # Iterative least-squares solution for receiver position and clock offset
        for _ in range(max_iterations):
            A = np.zeros((num_sats, 4))  # Design matrix
            omc = np.zeros(num_sats)     # Observed minus computed pseudoranges
            for i in range(num_sats):
                sat_pos_i = sat_positions[i]
                pr_i = pseudoranges[i]
                sat_clk_corr_i = sat_clock_corrections[i]
                delta_pos = sat_pos_i - current_pos
                geom_range = np.linalg.norm(delta_pos)
                # Correct pseudorange for satellite clock error
                pr_corrected = pr_i + C * sat_clk_corr_i
                omc[i] = pr_corrected - geom_range
                # Partial derivatives for design matrix
                A[i, 0] = -delta_pos[0] / geom_range
                A[i, 1] = -delta_pos[1] / geom_range
                A[i, 2] = -delta_pos[2] / geom_range
                A[i, 3] = 1.0  # Partial wrt receiver clock bias

            try:
                # Normal equation solution: delta_x = (A^T A)^-1 A^T omc
                N = A.T @ A
                N_inv = np.linalg.inv(N)
                delta_x = N_inv @ A.T @ omc
            except np.linalg.LinAlgError:
                # Singular matrix, cannot solve
                delta_x = None
                break

            # Update receiver position and clock bias
            current_pos += delta_x[:3]
            receiver_clock_offset_s = delta_x[3] / C  # Convert clock bias from meters to seconds

            # Check for convergence in position
            if np.linalg.norm(delta_x[:3]) < convergence_threshold:
                break

        if delta_x is not None:
            est_ecef = current_pos
            error_rinex = None
            if rinex_xyz is not None and len(rinex_xyz) == 3:
                # Compute error from RINEX header position (meters)
                error_rinex = float(np.linalg.norm(est_ecef - np.array(rinex_xyz)))
            # Store results for this epoch
            results.append({
                "epoch": epoch_time.isoformat(),
                "position_ecef": [float(est_ecef[0]), float(est_ecef[1]), float(est_ecef[2])],
                "receiver_clock_offset_ns": float(receiver_clock_offset_s * 1e9),
                "num_sats": int(num_sats),
                "error_from_rinex_header_m": error_rinex
            })
        processed_epochs += 1

    # Write results to JSON file
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    return results
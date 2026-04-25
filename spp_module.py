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

"""

import georinex as gr
import numpy as np
import xarray as xr
from datetime import datetime
import math
import logging
import json
from typing import Optional, List, Dict, Any, Tuple

# --- Constants ---
C = 299792458.0  # Speed of light in vacuum (meters per second)
GM = 3.986005e14  # WGS84 gravitational constant (m^3/s^2)
OMEGA_E_DOT = 7.2921151467e-5  # WGS84 Earth rotation rate (radians per second)
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

def ecef_to_geodetic(xyz: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECEF XYZ coordinates to WGS84 geodetic latitude, longitude, and height.

    Returns:
        Tuple of latitude (rad), longitude (rad), and ellipsoidal height (m).
    """
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - WGS84_E2))

    for _ in range(10):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)
        height = p / math.cos(lat) - n
        next_lat = math.atan2(z, p * (1 - WGS84_E2 * n / (n + height)))
        if abs(next_lat - lat) < 1e-12:
            lat = next_lat
            break
        lat = next_lat

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat**2)
    height = p / math.cos(lat) - n
    return lat, lon, height

def elevation_azimuth(receiver_xyz: np.ndarray, sat_xyz: np.ndarray) -> Tuple[float, float]:
    """
    Calculate satellite elevation and azimuth from receiver and satellite ECEF positions.

    Returns:
        Tuple of elevation (rad) and azimuth (rad).
    """
    lat, lon, _ = ecef_to_geodetic(receiver_xyz)
    dx = sat_xyz - receiver_xyz

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    east = -sin_lon * dx[0] + cos_lon * dx[1]
    north = -sin_lat * cos_lon * dx[0] - sin_lat * sin_lon * dx[1] + cos_lat * dx[2]
    up = cos_lat * cos_lon * dx[0] + cos_lat * sin_lon * dx[1] + sin_lat * dx[2]

    horizontal = math.hypot(east, north)
    elevation = math.atan2(up, horizontal)
    azimuth = math.atan2(east, north)
    if azimuth < 0:
        azimuth += 2 * math.pi
    return elevation, azimuth

def klobuchar_ionosphere_delay(
    receiver_xyz: np.ndarray,
    sat_xyz: np.ndarray,
    gps_sow: float,
    iono_coeffs: Optional[np.ndarray]
) -> float:
    """
    Calculate GPS L1 ionospheric group delay using the Klobuchar broadcast model.

    Returns:
        Delay in meters. Returns 0.0 when coefficients are unavailable.
    """
    if iono_coeffs is None or len(iono_coeffs) < 8:
        return 0.0

    alpha = np.asarray(iono_coeffs[:4], dtype=float)
    beta = np.asarray(iono_coeffs[4:8], dtype=float)
    lat, lon, _ = ecef_to_geodetic(receiver_xyz)
    elevation, azimuth = elevation_azimuth(receiver_xyz, sat_xyz)
    elevation_sc = max(elevation / math.pi, 0.0)
    lat_sc = lat / math.pi
    lon_sc = lon / math.pi
    azimuth_rad = azimuth

    psi = 0.0137 / (elevation_sc + 0.11) - 0.022
    phi_i = lat_sc + psi * math.cos(azimuth_rad)
    phi_i = min(max(phi_i, -0.416), 0.416)
    lam_i = lon_sc + psi * math.sin(azimuth_rad) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)

    local_time = (43200.0 * lam_i + gps_sow) % 86400.0
    amplitude = float(np.polyval(alpha[::-1], phi_m))
    period = float(np.polyval(beta[::-1], phi_m))
    amplitude = max(amplitude, 0.0)
    period = max(period, 72000.0)

    x = 2 * math.pi * (local_time - 50400.0) / period
    obliquity = 1.0 + 16.0 * (0.53 - elevation_sc)**3
    if abs(x) < 1.57:
        delay_s = obliquity * (5e-9 + amplitude * (1 - x**2 / 2 + x**4 / 24))
    else:
        delay_s = obliquity * 5e-9
    return C * delay_s

def saastamoinen_troposphere_delay(receiver_xyz: np.ndarray, sat_xyz: np.ndarray) -> float:
    """
    Estimate neutral-atmosphere delay with a simple Saastamoinen-style standard model.

    Returns:
        Delay in meters.
    """
    elevation, _ = elevation_azimuth(receiver_xyz, sat_xyz)
    sin_el = math.sin(max(elevation, math.radians(5.0)))
    _, _, height = ecef_to_geodetic(receiver_xyz)
    height = max(min(height, 10000.0), 0.0)

    pressure = 1013.25 * (1 - 2.2557e-5 * height) ** 5.2568
    temperature = 288.15 - 0.0065 * height
    water_vapor_pressure = 6.108 * 0.5 * math.exp((17.15 * (temperature - 273.15)) / (234.7 + (temperature - 273.15)))
    zenith_angle = math.pi / 2 - max(elevation, math.radians(5.0))

    return 0.002277 / sin_el * (
        pressure + (1255.0 / temperature + 0.05) * water_vapor_pressure - math.tan(zenith_angle) ** 2
    )

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

        # Relativistic correction (converted from meters)
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
            - "residuals_m": dictionary mapping satellite ID (str) to pseudorange residual (meters)
    """
    # Set up logging for info messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load RINEX observation and navigation files (GPS only for obs)
    obs = gr.load(obs_file, use='G')
    nav = gr.load(nav_file)
    if pseudorange_code not in obs.data_vars:
        available = ", ".join(str(name) for name in obs.data_vars)
        raise ValueError(f"pseudorange code '{pseudorange_code}' not found in observation file. Available: {available}")

    # Get receiver's approximate position from RINEX header, if available
    rinex_xyz = None
    try:
        rinex_xyz = obs.attrs.get('position_xyz')
        if rinex_xyz is None:
            rinex_xyz = obs.attrs.get('position')
        if isinstance(rinex_xyz, np.ndarray):
            rinex_xyz = rinex_xyz.tolist()
        elif isinstance(rinex_xyz, (xr.DataArray, xr.Variable)):
            rinex_xyz = rinex_xyz.values.tolist()
    except Exception:
        rinex_xyz = None
    if rinex_xyz is None:
        rinex_xyz = parse_rinex_header_xyz(obs_file)

    iono_coeffs = nav.attrs.get('ionospheric_corr_GPS')

    # Prefer the RINEX approximate position; fall back to Dallas, TX if unavailable.
    if initial_xyz is None:
        if rinex_xyz is not None and len(rinex_xyz) == 3:
            initial_xyz = rinex_xyz
        else:
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
        used_svs = []

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
            # Estimate transmit time by correcting reception time for signal travel time
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
                used_svs.append(sv)

        if len(sat_positions) < min_sats:
            continue  # Not enough satellites for a solution

        num_sats = len(sat_positions)
        # Removed unused state vector initialization 'x'
        current_pos = np.array(initial_xyz)
        delta_x = None
        receiver_clock_bias_m = 0.0

        # Iterative least-squares solution for receiver position and clock offset
        for iter_idx in range(max_iterations):
            A = np.zeros((num_sats, 4))  # Design matrix
            omc = np.zeros(num_sats)     # Observed minus computed pseudoranges
            for i in range(num_sats):
                sat_pos_i = sat_positions[i]
                pr_i = pseudoranges[i]
                sat_clk_corr_i = sat_clock_corrections[i]
                delta_pos = sat_pos_i - current_pos
                geom_range = np.linalg.norm(delta_pos)
                iono_delay = klobuchar_ionosphere_delay(current_pos, sat_pos_i, time_of_week, iono_coeffs)
                tropo_delay = saastamoinen_troposphere_delay(current_pos, sat_pos_i)
                # Correct pseudorange for satellite clock, ionosphere, and troposphere.
                pr_corrected = pr_i + C * sat_clk_corr_i - iono_delay - tropo_delay
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
            receiver_clock_bias_m = float(delta_x[3])
            receiver_clock_offset_s = receiver_clock_bias_m / C  # Convert clock bias from meters to seconds

            # Compute convergence metric
            pos_update_norm = float(np.linalg.norm(delta_x[:3]))

            # Check for convergence in position
            if pos_update_norm < convergence_threshold:
                break
        # --- Iteration loop finished --- 

        if delta_x is not None:
            
            est_ecef = current_pos
            
            # --- Calculate final residuals --- 
            final_omc = np.zeros(num_sats)
            sv_labels = [str(sv) for sv in used_svs] # Ensure labels match satellites actually used
            
            for i in range(num_sats):
                sat_pos_i = sat_positions[i]
                pr_i = pseudoranges[i]
                sat_clk_corr_i = sat_clock_corrections[i]
                delta_pos = sat_pos_i - est_ecef
                geom_range = np.linalg.norm(delta_pos)
                iono_delay = klobuchar_ionosphere_delay(est_ecef, sat_pos_i, time_of_week, iono_coeffs)
                tropo_delay = saastamoinen_troposphere_delay(est_ecef, sat_pos_i)
                pr_corrected = pr_i + C * sat_clk_corr_i - iono_delay - tropo_delay
                final_omc[i] = pr_corrected - geom_range - receiver_clock_bias_m
            
            # Map residuals to SV labels
            residuals_dict = {sv_labels[i]: float(final_omc[i]) for i in range(num_sats)}
            
            # Calculate error from RINEX header
            error_rinex = None
            if rinex_xyz is not None and len(rinex_xyz) == 3:
                error_rinex = float(np.linalg.norm(est_ecef - np.array(rinex_xyz)))
            
            # Store results
            results.append({
                "epoch": epoch_time.isoformat(),
                "position_ecef": [float(est_ecef[0]), float(est_ecef[1]), float(est_ecef[2])],
                "receiver_clock_offset_ns": float(receiver_clock_offset_s * 1e9),
                "num_sats": int(num_sats),
                "error_from_rinex_header_m": error_rinex,
                "residuals_m": residuals_dict
            })
            processed_epochs += 1

    # Write results to JSON file
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    return results

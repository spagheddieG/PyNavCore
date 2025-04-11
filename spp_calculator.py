import georinex as gr
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import math
import logging

# --- Configuration ---
OBS_FILE = 'algo0010.25o'
NAV_FILE = 'brdc0010.25n'
# Use C1 pseudorange (as C1C is not present in this RINEX v2 file)
PSEUDORANGE_CODE = 'C1' 
# Number of epochs to process (set to None to process all)
MAX_EPOCHS = 5 
# Minimum number of satellites required for a position fix
MIN_SATS = 4 
# Convergence threshold for iterative least squares
CONVERGENCE_THRESHOLD = 1e-4
MAX_ITERATIONS = 10

# --- Constants ---
C = 299792458.0  # Speed of light (m/s)
GM = 3.986005e14  # WGS84 gravitational constant (m^3/s^2) - mu in IS-GPS-200
OMEGA_E_DOT = 7.2921151467e-5  # WGS84 Earth rotation rate (rad/s)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def calculate_satellite_position_and_clock(ephem, transmit_time_gps_week, transmit_time_sow):
    """
    Calculates satellite position and clock correction from broadcast ephemeris.
    Based on IS-GPS-200.

    Args:
        ephem (xarray.Dataset): Ephemeris data for a single satellite and time.
        transmit_time_gps_week (int): GPS week of transmission time.
        transmit_time_sow (float): Seconds of week of transmission time.

    Returns:
        tuple: (sat_pos_ecef, sat_clock_corr) or (None, None) if calculation fails.
               sat_pos_ecef: numpy array [X, Y, Z] satellite position in ECEF (m)
               sat_clock_corr: satellite clock correction (s)
    """
    try:
        # Time calculation
        toe = ephem['Toe'].item() # Time of ephemeris (seconds of GPS week)
        # Use Toe as a substitute for Toc if Toc is missing
        try:
            toc = ephem['Toc'].item() # Time of clock (seconds of GPS week)
        except KeyError:
            toc = ephem['Toe'].item() # Fallback: use Toe if Toc is missing
            logging.warning("Toc not found in ephemeris; using Toe as substitute for clock correction.")
        
        # Ensure transmit time is relative to the ephemeris reference week if necessary
        # (For simplicity, assume transmit_time_sow is already in the correct week context)
        tk = transmit_time_sow - toe 
        # Account for week rollovers
        if tk > 302400:
            tk -= 604800
        elif tk < -302400:
            tk += 604800

        # --- Satellite Clock Correction ---
        af0 = ephem['SVclockBias'].item()
        af1 = ephem['SVclockDrift'].item()
        af2 = ephem['SVclockDriftRate'].item()
        
        # Time difference for clock correction
        dt_clock = transmit_time_sow - toc
        if dt_clock > 302400:
            dt_clock -= 604800
        elif dt_clock < -302400:
            dt_clock += 604800

        sat_clock_bias = af0 + af1 * dt_clock + af2 * dt_clock**2

        # --- Satellite Position Calculation ---
        a = ephem['sqrtA'].item()**2  # Semi-major axis
        n0 = math.sqrt(GM / a**3)  # Computed mean motion
        n = n0 + ephem['DeltaN'].item()  # Corrected mean motion
        
        Mk = ephem['M0'].item() + n * tk  # Mean anomaly

        # Iteratively solve Kepler's equation for eccentric anomaly (Ek)
        Ek = Mk
        for _ in range(10): # Max 10 iterations
            Ek_old = Ek
            Ek = Mk + ephem['Eccentricity'].item() * math.sin(Ek_old)
            if abs(Ek - Ek_old) < 1e-12:
                break
        else:
            logging.warning(f"Kepler's equation did not converge for SV {ephem['sv'].item()}")
            return None, None # Convergence failure

        # True anomaly (vk)
        sin_Ek = math.sin(Ek)
        cos_Ek = math.cos(Ek)
        e = ephem['Eccentricity'].item()
        vk_num = math.sqrt(1 - e**2) * sin_Ek
        vk_den = cos_Ek - e
        vk = math.atan2(vk_num, vk_den)

        # Argument of latitude (Phik)
        Phik = vk + ephem['omega'].item()

        # Second harmonic perturbations
        sin2Phik = math.sin(2 * Phik)
        cos2Phik = math.cos(2 * Phik)
        duk = ephem['Cuc'].item() * cos2Phik + ephem['Cus'].item() * sin2Phik  # Argument of latitude correction
        drk = ephem['Crc'].item() * cos2Phik + ephem['Crs'].item() * sin2Phik  # Radius correction
        dik = ephem['Cic'].item() * cos2Phik + ephem['Cis'].item() * sin2Phik  # Inclination correction

        uk = Phik + duk  # Corrected argument of latitude
        rk = a * (1 - e * cos_Ek) + drk  # Corrected radius
        ik = ephem['Io'].item() + ephem['IDOT'].item() * tk + dik  # Corrected inclination

        # Position in orbital plane
        xk_prime = rk * math.cos(uk)
        yk_prime = rk * math.sin(uk)

        # Corrected longitude of ascending node
        Omega_k = ephem['Omega0'].item() + (ephem['OmegaDot'].item() - OMEGA_E_DOT) * tk - OMEGA_E_DOT * toe

        # ECEF coordinates
        cos_Omega_k = math.cos(Omega_k)
        sin_Omega_k = math.sin(Omega_k)
        cos_ik = math.cos(ik)
        sin_ik = math.sin(ik)

        Xk = xk_prime * cos_Omega_k - yk_prime * cos_ik * sin_Omega_k
        Yk = xk_prime * sin_Omega_k + yk_prime * cos_ik * cos_Omega_k
        Zk = yk_prime * sin_ik

        sat_pos_ecef = np.array([Xk, Yk, Zk])

        # --- Relativistic Clock Correction ---
        # Note: Simplified version, full correction depends on receiver position too
        relativistic_corr = -2 * math.sqrt(GM * a) * e * sin_Ek / C**2
        # F = -2 * sqrt(mu) / c^2 = -4.442807633e-10 [s/m^(1/2)] IS-GPS-200 Eq. 20.3.3.3.3.1-5
        # relativistic_corr = -4.442807633e-10 * e * ephem['sqrtA'].item() * sin_Ek

        sat_clock_corr = sat_clock_bias + relativistic_corr - ephem['TGD'].item() # TGD for single freq users

        return sat_pos_ecef, sat_clock_corr

    except Exception as e:
        logging.error(f"Error calculating position/clock for SV {ephem.get('sv', 'Unknown')}: {e}")
        return None, None

# --- Main SPP Function ---
def perform_spp(obs_data, nav_data, approx_pos):
    """
    Performs Single Point Positioning for epochs in the observation data.

    Args:
        obs_data (xarray.Dataset): Parsed observation data.
        nav_data (xarray.Dataset): Parsed navigation data.
        approx_pos (np.array): Approximate receiver position [X, Y, Z] in ECEF (m).

    Returns:
        list: A list of dictionaries, each containing epoch time and calculated position.
    """
    results = []
    processed_epochs = 0

    # Ensure 'time' is a coordinate for easier selection
    if 'time' not in obs_data.coords:
         obs_data = obs_data.set_coords('time')
         
    # Ensure 'sv' is a coordinate in nav_data
    if 'sv' not in nav_data.coords:
        nav_data = nav_data.set_coords('sv')

    # Iterate through observation epochs
    unique_times = np.unique(obs_data['time'].values)
    logging.info(f"Found {len(unique_times)} unique epochs in OBS file.")

    for epoch_time_dt64 in unique_times:
        if MAX_EPOCHS is not None and processed_epochs >= MAX_EPOCHS:
            break
        
        epoch_time = gr.to_datetime(epoch_time_dt64) # Convert numpy.datetime64 to datetime
        logging.info(f"\nProcessing epoch: {epoch_time}")

        # Select observations for the current epoch
        epoch_obs = obs_data.sel(time=epoch_time_dt64)

        # Filter satellites with the required pseudorange observation
        valid_svs = epoch_obs['sv'][np.isfinite(epoch_obs[PSEUDORANGE_CODE])].values
        if len(valid_svs) < MIN_SATS:
            logging.warning(f"Skipping epoch {epoch_time}: Only {len(valid_svs)} satellites with '{PSEUDORANGE_CODE}' observations (minimum {MIN_SATS} required).")
            continue
        
        epoch_obs = epoch_obs.sel(sv=valid_svs)
        logging.debug(f"Satellites for this epoch: {epoch_obs['sv'].values}")

        # --- Prepare data for least squares ---
        sat_positions = []
        sat_clock_corrections = []
        pseudoranges = []
        sv_list_for_epoch = []

        # Get GPS week and seconds of week for the current epoch
        # Need to handle potential week rollovers if obs spans multiple weeks
        # For simplicity, assume all obs are within one week or georinex handles it
        # Calculate GPS week and seconds of week manually
        GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0) # GPS epoch start
        # Ensure epoch_time is a Python datetime object (convert if needed)
        if isinstance(epoch_time, np.datetime64):
            epoch_time = gr.to_datetime(epoch_time)
        # If still not a datetime.datetime, try to convert using pandas
        if not isinstance(epoch_time, datetime):
            try:
                import pandas as pd
                epoch_time = pd.to_datetime(epoch_time).to_pydatetime()
            except Exception:
                raise TypeError(f"Could not convert epoch_time to datetime.datetime, got {type(epoch_time)}")
        # Now epoch_time should be a datetime.datetime object (usually naive UTC)
        # If it has tzinfo, make it naive
        if hasattr(epoch_time, "tzinfo") and epoch_time.tzinfo is not None:
            epoch_time = epoch_time.replace(tzinfo=None)
        time_diff = epoch_time - GPS_EPOCH
        total_seconds = time_diff.total_seconds()
        # Note: This doesn't account for GPS leap seconds vs UTC leap seconds,
        # but should be close enough for broadcast ephemeris selection.
        # A more robust solution would use a dedicated time library.
        gps_week = int(total_seconds // (7 * 24 * 3600))
        time_of_week = total_seconds % (7 * 24 * 3600)

        for sv in epoch_obs['sv'].values:
            pr = epoch_obs[PSEUDORANGE_CODE].sel(sv=sv).item()
            
            # Find the closest ephemeris in time *before* the observation time
            # Use the time_of_week for comparison with Toe
            available_ephem = nav_data.sel(sv=sv, time=slice(None, epoch_time_dt64)).dropna(dim='time', how='all')
            if available_ephem.time.size == 0:
                 logging.warning(f"No ephemeris found for SV {sv} at or before {epoch_time}. Skipping satellite.")
                 continue
            
            # Select the latest ephemeris before or at the observation time
            ephem = available_ephem.isel(time=-1) 
            
            # Estimate transmission time (reception time - propagation time)
            # Initial guess for propagation time using approximate position
            approx_dist = np.linalg.norm(approx_pos) # Rough distance from Earth center
            prop_time_guess = approx_dist / C 
            transmit_time_sow = time_of_week - pr / C # Corrected for pseudorange
            
            # Calculate satellite position and clock correction at estimated transmit time
            sat_pos, sat_clk = calculate_satellite_position_and_clock(ephem, gps_week, transmit_time_sow)

            if sat_pos is not None and sat_clk is not None:
                 # Account for Earth rotation during signal travel time (Sagnac effect)
                 # Rotation angle
                 omega_tau = OMEGA_E_DOT * (pr / C) 
                 # Rotation matrix
                 Rz = np.array([[ math.cos(omega_tau), math.sin(omega_tau), 0],
                                [-math.sin(omega_tau), math.cos(omega_tau), 0],
                                [ 0,                0,               1]])
                 # Apply rotation to satellite position
                 sat_pos_corrected = Rz @ sat_pos 

                 sat_positions.append(sat_pos_corrected)
                 sat_clock_corrections.append(sat_clk)
                 pseudoranges.append(pr)
                 sv_list_for_epoch.append(sv)
            else:
                 logging.warning(f"Could not compute position/clock for SV {sv} at {epoch_time}. Skipping satellite.")


        if len(sat_positions) < MIN_SATS:
            logging.warning(f"Skipping epoch {epoch_time}: Only {len(sat_positions)} valid satellite positions calculated (minimum {MIN_SATS} required).")
            continue

        # --- Iterative Least Squares ---
        num_sats = len(sat_positions)
        logging.info(f"Performing SPP for epoch {epoch_time} with {num_sats} satellites: {sv_list_for_epoch}")
        
        # Initial state vector: [dx, dy, dz, dt*C] (corrections to approx pos + receiver clock offset * C)
        x = np.zeros(4) 
        current_pos = approx_pos.copy() # Use a copy for iteration
        
        for iteration in range(MAX_ITERATIONS):
            A = np.zeros((num_sats, 4)) # Design matrix
            omc = np.zeros(num_sats)    # Observed minus computed pseudorange vector

            for i in range(num_sats):
                sat_pos_i = sat_positions[i]
                pr_i = pseudoranges[i]
                sat_clk_corr_i = sat_clock_corrections[i]

                # Geometric range calculation
                delta_pos = sat_pos_i - current_pos
                geom_range = np.linalg.norm(delta_pos)

                # Observed minus computed pseudorange (omc = PR - (range - c*dt_sv + c*dt_rec))
                # We solve for dt_rec*C, so the equation becomes:
                # PR_corrected = PR + c*dt_sv 
                # omc = PR_corrected - geom_range = c*dt_rec + error
                pr_corrected = pr_i + C * sat_clk_corr_i
                omc[i] = pr_corrected - geom_range

                # Fill design matrix (partial derivatives)
                A[i, 0] = -delta_pos[0] / geom_range # d(range)/dx
                A[i, 1] = -delta_pos[1] / geom_range # d(range)/dy
                A[i, 2] = -delta_pos[2] / geom_range # d(range)/dz
                A[i, 3] = 1.0                         # d(range)/d(c*dt_rec)

            # Least squares solution: dx = (A^T * A)^-1 * A^T * omc
            try:
                N = A.T @ A # Normal matrix
                N_inv = np.linalg.inv(N)
                delta_x = N_inv @ A.T @ omc
            except np.linalg.LinAlgError:
                logging.error(f"Matrix inversion failed at epoch {epoch_time}, iteration {iteration}. Skipping epoch.")
                delta_x = None # Indicate failure
                break # Break from iteration loop

            # Update receiver position and clock offset estimate
            current_pos += delta_x[:3]
            # The 4th element is c * dt_rec, store dt_rec
            receiver_clock_offset_s = delta_x[3] / C 

            # Check for convergence
            pos_change_norm = np.linalg.norm(delta_x[:3])
            logging.debug(f"Iteration {iteration+1}: Position change norm = {pos_change_norm:.4f} m, Clock offset = {receiver_clock_offset_s*1e9:.2f} ns")
            if pos_change_norm < CONVERGENCE_THRESHOLD:
                logging.info(f"Converged after {iteration+1} iterations.")
                break # Converged
        else:
             # Loop finished without break (no convergence)
             logging.warning(f"SPP did not converge within {MAX_ITERATIONS} iterations for epoch {epoch_time}.")
             delta_x = None # Indicate failure

        if delta_x is not None: # Check if least squares succeeded and converged
            final_pos = current_pos
            results.append({
                'time': epoch_time,
                'X': final_pos[0],
                'Y': final_pos[1],
                'Z': final_pos[2],
                'dt_rec_ns': receiver_clock_offset_s * 1e9, # Receiver clock offset in ns
                'num_sats': num_sats
            })
            # Update approx_pos for the next epoch for potentially faster convergence
            approx_pos = final_pos 
            
        processed_epochs += 1

    return results

# --- Main Execution ---
if __name__ == "__main__":
    try:
        logging.info(f"Loading OBS file: {OBS_FILE}")
        obs = gr.load(OBS_FILE, use='G') # Load only GPS observations

        logging.info(f"Loading NAV file: {NAV_FILE}")
        nav = gr.load(NAV_FILE)

        # Extract approximate position from OBS header - try georinex first, then manual parse
        approx_xyz = None
        # Dallas, Texas approximate ECEF coordinates (WGS84): X= -1288392.5, Y= -4865182.1, Z= 3999769.7 (meters)
        approx_xyz = [-1288392.5, -4865182.1, 3999769.7]
        logging.info(f"Using hardcoded Dallas, TX ECEF position as initial estimate: {approx_xyz}")
        # If you want to use the OBS file's header instead, comment out the above and uncomment below:
        # try:
        #     approx_xyz = obs.attrs['position_xyz']
        #     logging.info(f"Using approximate position from georinex attrs: {approx_xyz}")
        #         with open(OBS_FILE, 'r') as f:
        #             for line in f:
        #                 if "APPROX POSITION XYZ" in line:
        #                     parts = line.split()
        #                     if len(parts) >= 4:
        #                         approx_xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
        #                         logging.info(f"Manually parsed approximate position: {approx_xyz}")
        #                         break
        #                     else:
        #                         logging.warning(f"Found 'APPROX POSITION XYZ' line but couldn't parse coordinates: {line.strip()}")
        #         if approx_xyz is None:
        #             raise ValueError("Manual parsing failed to find 'APPROX POSITION XYZ' line or parse coordinates.")
        #     except Exception as e:
        #         logging.error(f"Could not determine approximate position from OBS header: {e}. Exiting.")
        #         exit()
            
        # Perform SPP
        logging.info("Starting SPP calculation...")
        spp_results = perform_spp(obs, nav, np.array(approx_xyz))

        # Print results
        print("\n--- SPP Results ---")
        if spp_results:
            # Try to get the RINEX header position for error calculation
            rinex_xyz = None
            # Parse the RINEX header directly if not available in obs.attrs
            try:
                rinex_xyz = obs.attrs['position_xyz']
                if isinstance(rinex_xyz, np.ndarray):
                    rinex_xyz = rinex_xyz.tolist()
                elif isinstance(rinex_xyz, (xr.DataArray, xr.Variable)):
                    rinex_xyz = rinex_xyz.values.tolist()
            except Exception:
                # Manual parse as fallback
                try:
                    with open(OBS_FILE, 'r') as f:
                        for line in f:
                            if "APPROX POSITION XYZ" in line:
                                parts = line.split()
                                if len(parts) >= 4:
                                    rinex_xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
                                break
                except Exception:
                    rinex_xyz = None
            for res in spp_results:
                print(f"Epoch: {res['time']}")
                print(f"  Position (X, Y, Z) ECEF: {res['X']:.3f} m, {res['Y']:.3f} m, {res['Z']:.3f} m")
                print(f"  Receiver Clock Offset: {res['dt_rec_ns']:.2f} ns")
                print(f"  Satellites Used: {res['num_sats']}")
                # Calculate and display error from initial position estimate (Dallas, TX)
                dallas_ecef = np.array([-1288392.5, -4865182.1, 3999769.7])
                est_ecef = np.array([res['X'], res['Y'], res['Z']])
                error_dallas = np.linalg.norm(est_ecef - dallas_ecef)
                print(f"  Error from Dallas, TX initial estimate: {error_dallas:.3f} m")
                # Calculate and display error from RINEX header position if available
                if rinex_xyz is not None and len(rinex_xyz) == 3:
                    error_rinex = np.linalg.norm(est_ecef - np.array(rinex_xyz))
                    print(f"  Error from RINEX header position: {error_rinex:.3f} m")
        else:
            print("No valid positions calculated.")

        # If only one error is printed, clarify why
        if spp_results:
            if rinex_xyz is None or not (isinstance(rinex_xyz, (list, np.ndarray)) and len(rinex_xyz) == 3):
                print("Note: RINEX header position not found or invalid, so only Dallas, TX error is shown.")
            else:
                print("Both errors (from Dallas, TX and RINEX header) are shown above for each epoch.")

    except FileNotFoundError as e:
        logging.error(f"Error loading file: {e}. Make sure RINEX files are in the same directory or provide full paths.")
    except ImportError as e:
        logging.error(f"Missing required library: {e}. Please install using 'pip install georinex numpy scipy'")
    # Remove stray except/indentation from commented block above
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
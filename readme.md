# spp_module.py

## Overview

`spp_module.py` is a modular Python implementation of a Single Point Positioning (SPP) solver for GNSS (Global Navigation Satellite System) data. It processes RINEX observation and navigation files to estimate the receiver's position and clock offset using broadcast ephemeris and pseudorange measurements. The module is designed for clarity, extensibility, and educational use, making it suitable for research, prototyping, and teaching GNSS fundamentals.

## Features

- Computes satellite ECEF positions and clock corrections from broadcast ephemeris (GPS ICD-200 compliant).
- Parses receiver's approximate position from RINEX observation file headers.
- Iteratively solves for receiver position and clock offset using least-squares adjustment.
- Outputs results (epoch, position, clock offset, error from header, number of satellites) to a JSON file.
- Well-commented and structured for easy understanding and modification.

## Dependencies

- [georinex](https://github.com/geospace-code/georinex): For reading RINEX observation and navigation files.
- [numpy](https://numpy.org/): Numerical operations.
- [xarray](https://xarray.dev/): Array and dataset handling.
- Standard Python libraries: `datetime`, `math`, `logging`, `json`, `typing`.

## Usage

```python
from spp_module import spp_solve

results = spp_solve(
    obs_file="your_obs_file.##o",
    nav_file="your_nav_file.##n",
    pseudorange_code="C1",           # or "C1C", "P1", etc. depending on your data
    max_epochs=10,                   # process up to 10 epochs
    min_sats=4,                      # minimum satellites required
    convergence_threshold=1e-4,      # meters
    max_iterations=10,               # max iterations per epoch
    initial_xyz=None,                # use RINEX header or default if None
    output_json="spp_results.json"   # output file
)
```

## Main Functions

- **calculate_satellite_position_and_clock**: Computes satellite ECEF position and clock correction for a given transmit time using broadcast ephemeris.
- **parse_rinex_header_xyz**: Extracts the receiver's approximate ECEF position from the RINEX observation file header.
- **spp_solve**: Main entry point. Loads RINEX files, iterates over epochs, selects valid satellites, computes satellite positions, and solves for receiver position and clock offset using least-squares.

## Workflow

1. Load RINEX observation and navigation files.
2. For each epoch:
   - Identify satellites with valid pseudorange measurements.
   - For each satellite, select the latest valid ephemeris and compute its ECEF position and clock correction.
   - Apply Sagnac correction for Earth's rotation.
   - Formulate and solve the least-squares problem for receiver position and clock offset.
   - Store results and compute error from RINEX header position (if available).
3. Write results to a JSON file.

## Proposed Enhancements

To increase the functionality and accuracy of this module, we are considering the following improvements:

| Enhancement                                      | Status       |
|--------------------------------------------------|--------------|
| Multi-Constellation Support                      | Not Started  |
| Advanced Error Modeling (iono/tropo, antenna PC) | Not Started  |
| Robust Outlier Detection                         | Not Started  |
| Improved Convergence and Initialization          | Not Started  |
| Dilution of Precision (DOP) Calculation          | Not Started  |
| Enhanced Output and Visualization                | Not Started  |
| Real-Time/Streaming Capability                   | Not Started  |
| Code Structure and Performance Improvements      | Not Started  |
| Core SPP Functionality (GPS)                     | Complete     |

### 1. Multi-Constellation Support
- Extend support to other GNSS constellations (GLONASS, Galileo, BeiDou) by handling their navigation message formats and coordinate systems.

### 2. Advanced Error Modeling
- Implement tropospheric and ionospheric delay corrections using standard models (e.g., Klobuchar, Saastamoinen).
- Add support for satellite and receiver antenna phase center corrections.
- Incorporate multipath mitigation techniques.

### 3. Robust Outlier Detection
- Add statistical tests (e.g., residual analysis, RANSAC) to detect and exclude outlier measurements or faulty satellites.

### 4. Improved Convergence and Initialization
- Use more sophisticated initial position estimation (e.g., weighted centroid of satellites).
- Implement adaptive convergence thresholds and iteration limits based on data quality.

### 5. Dilution of Precision (DOP) Calculation
- Compute and report DOP values (GDOP, PDOP, HDOP, VDOP) for each epoch to assess solution quality.

### 6. Enhanced Output and Visualization
- Add support for outputting results in standard formats (CSV, KML, etc.).
- Provide plotting utilities for position errors, satellite geometry, and residuals.

### 7. Real-Time/Streaming Capability
- Adapt the solver for real-time or near-real-time processing of GNSS data streams.

### 8. Code Structure and Performance
- Refactor for object-oriented design (e.g., a `SPPSolver` class).
- Optimize performance for large datasets (e.g., vectorized operations, parallel processing).

## GNSS Constellation Support

| Constellation | Supported | Notes                        |
|---------------|-----------|------------------------------|
| GPS           | Yes       | Fully supported              |
| GLONASS       | No        | Not Planned                  |
| Galileo       | No        | Not Planned                  |
| BeiDou        | No        | Not Planned                  |

## References

- GPS Interface Control Document ICD-GPS-200 (latest revision)
- [georinex documentation](https://georinex.readthedocs.io/)
- GNSS textbooks

---

For questions or contributions, please open an issue or submit a pull request.
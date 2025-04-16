# Single Point Positioning (SPP) Solver

## Overview

This project implements a Single Point Positioning (SPP) solver for GNSS (Global Navigation Satellite System) data, with a focus on GPS processing. The implementation processes RINEX observation and navigation files to estimate receiver position and clock offset using broadcast ephemeris and pseudorange measurements.

## Features

### Core Functionality
- Processes RINEX observation and navigation files
- Computes satellite ECEF positions and clock corrections using broadcast ephemeris
- Implements iterative least-squares positioning algorithm
- Handles Sagnac correction for Earth rotation effects
- Supports multiple epochs processing with configurable limits
- Provides detailed convergence logging and analysis
- Outputs results in JSON format for further analysis

### Implementation Details
- Automatic initial position estimation from RINEX header
- Fallback to default position (Dallas, TX) when header position unavailable
- Configurable convergence criteria and iteration limits
- Satellite selection based on valid pseudorange measurements
- Comprehensive error checking and validation
- Progress tracking with tqdm for long processing runs

## Dependencies

- georinex: RINEX file parsing
- numpy: Numerical operations and linear algebra
- xarray: Multi-dimensional data handling
- tqdm: Progress bar visualization
- Standard Python libraries: datetime, math, logging, json, typing

## Usage

```python
from spp_module import spp_solve

results = spp_solve(
    obs_file="your_obs_file.##o",
    nav_file="your_nav_file.##n",
    pseudorange_code="C1",           # or "C1C", "P1", etc. depending on data
    max_epochs=10,                   # process up to 10 epochs
    min_sats=4,                      # minimum satellites required
    convergence_threshold=1e-4,      # meters
    max_iterations=10,               # max iterations per epoch
    initial_xyz=None,                # use RINEX header or default if None
    output_json="spp_results.json",  # main results output
    convergence_log_json="spp_convergence_log.json"  # detailed convergence data
)
```

## Output Files

### spp_results.json
Contains per-epoch results including:
- Epoch timestamp (ISO8601 format)
- ECEF position [X, Y, Z] in meters
- Receiver clock offset in nanoseconds
- Number of satellites used
- Error from RINEX header position (if available)

### spp_convergence_log.json
Detailed convergence information including:
- Per-iteration position updates
- Pseudorange residuals
- Geometric ranges
- Convergence metrics
- Satellite-specific data

## Development Status

| Feature                                        | Status    |
|-----------------------------------------------|-----------|
| Core SPP Implementation                        | Complete  |
| GPS Constellation Support                      | Complete  |
| Progress Visualization                         | Complete  |
| Convergence Analysis                          | Complete  |
| Results Visualization                         | Complete  |
| Multi-Constellation Support                   | Planned   |
| Tropospheric/Ionospheric Corrections         | Planned   |
| Outlier Detection                            | Planned   |
| Real-Time Processing                         | Planned   |

## Visualization Tools

The project includes a visualizer module that provides:
- Position error plotting over time
- Pseudorange residuals analysis
- Satellite geometry visualization
- Convergence metrics plotting

## Future Enhancements

1. **Advanced Error Modeling**
   - Tropospheric delay modeling
   - Ionospheric corrections
   - Antenna phase center variations

2. **Robustness Improvements**
   - RANSAC-based outlier detection
   - Advanced initialization techniques
   - Adaptive convergence thresholds

3. **Performance Optimization**
   - Vectorized operations
   - Parallel processing support
   - Memory optimization for large datasets

4. **Extended Output Formats**
   - KML/KMZ for Google Earth
   - RINEX output capability
   - Standard geodetic formats

## References

- GPS Interface Control Document (ICD-GPS-200)
- [georinex documentation](https://georinex.readthedocs.io/)
- GNSS textbooks and literature

For questions, bug reports, or contributions, please open an issue or submit a pull request.
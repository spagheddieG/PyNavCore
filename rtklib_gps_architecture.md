# RTKLIB-like GPS Positioning Script: High-Level Architecture

## 1. Main Modules, Classes, and Functions

### rinex
- **RinexHeaderParser**: Parses RINEX observation and navigation file headers.
- **RinexDataParser**: Parses RINEX observation and navigation data records.
- **RinexObservation, RinexNavigation**: Data classes for storing parsed observation and navigation data.

### ephemeris
- **EphemerisExtractor**: Extracts and manages satellite ephemeris from navigation data.
- **Ephemeris**: Data class for satellite ephemeris parameters.

### observation
- **ObservationExtractor**: Extracts and manages observation data (pseudorange, carrier phase, etc.).
- **ObservationEpoch**: Data class for a single epoch’s observations.

### pseudorange
- **PseudorangeCalculator**: Computes pseudoranges for each satellite, applies corrections (clock bias, etc.).

### positioning
- **PositionSolver**: Calculates receiver position using pseudoranges and ephemeris (SPP algorithm).
- **PositionSolution**: Data class for position results, including quality indicators.

### system
- **SatelliteSystemManager**: Identifies and manages different GNSS systems (GPS, GLO, Galileo, BeiDou).
- **SystemConfig**: Configuration for supported/active systems.

### io
- **ResultFormatter**: Formats output as CSV and JSON.
- **Logger**: Handles warnings, errors, and info messages.

### cli
- **main()**: Entry point for CLI usage.
- **ArgumentParser**: Handles command-line arguments.

---

## 2. Data Flow Overview

```mermaid
flowchart TD
    A[RINEX Obs/Nav Files] --> B[RinexHeaderParser]
    B --> C[RinexDataParser]
    C --> D[EphemerisExtractor]
    C --> E[ObservationExtractor]
    D --> F[PseudorangeCalculator]
    E --> F
    F --> G[PositionSolver]
    G --> H[ResultFormatter]
    H --> I[Output (CSV/JSON)]
    subgraph Error/Warning Handling
        J[Logger]
    end
    B -- errors/warnings --> J
    C -- errors/warnings --> J
    D -- errors/warnings --> J
    E -- errors/warnings --> J
    F -- errors/warnings --> J
    G -- errors/warnings --> J
```

---

## 3. Key Algorithms

- **Ephemeris Extraction**: Parse navigation messages, extract satellite orbits, clock corrections, and system-specific parameters.
- **Pseudorange Computation**: Use correct observation type (e.g., C1), apply clock bias and (optionally) ionospheric corrections.
- **Position Solution (SPP)**: Use least-squares or iterative method to solve for receiver position (WGS84), require at least 4 satellites.
- **System Handling**: Identify satellite system from RINEX, use correct ephemeris/observation types, allow for easy extension to new systems.

---

## 4. Handling Mixed Satellite Systems and Edge Cases

- **SatelliteSystemManager** abstracts system-specific logic (e.g., ephemeris formats, observation types).
- **Graceful Degradation**: If unsupported system or observation type is encountered, skip and log a warning.
- **Error Handling**: Detect and report missing data, malformed records, or insufficient satellites for solution.
- **Extensibility**: New systems (GLO, Galileo, BeiDou) can be added by extending SatelliteSystemManager and ephemeris/observation parsing logic.

---

## 5. Extensibility and Maintainability Recommendations

- **Modular Design**: Each major function (parsing, computation, output) is a separate module/class.
- **Data Classes**: Use Python dataclasses for structured data (ephemeris, observations, solutions).
- **Configurable System Support**: SystemConfig allows enabling/disabling GNSS systems.
- **Clear Interfaces**: Each module exposes clear, testable interfaces for TDD.
- **Error/Warning Logging**: Centralized Logger for all error/warning/info messages.
- **Output Abstraction**: ResultFormatter supports multiple output formats; easy to add more.
- **CLI and Library**: main() provides CLI entry, but all logic is reusable as a library.

---

## 6. Example Directory Structure

```
rtklib_gps/
├── rinex.py
├── ephemeris.py
├── observation.py
├── pseudorange.py
├── positioning.py
├── system.py
├── io.py
├── cli.py
├── __init__.py
```

---

## 7. TDD Guidance

- Each Gherkin scenario maps to a test module.
- Each module/class exposes methods that can be unit/integration tested.
- Use dependency injection for system configuration and logging to facilitate mocking in tests.

---
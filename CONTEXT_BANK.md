in# Context Bank

---

## 2025-04-11

### RTKLIB-like GPS Positioning Script: Architecture, TDD Process, and Module Overview

#### Rationale for Modular Architecture and Refactor Decisions

The project adopts a modular architecture to ensure clarity, maintainability, and extensibility. Each major function—parsing, computation, output formatting, and system management—is encapsulated in a dedicated module or class. This separation of concerns allows for:
- Easier unit and integration testing (TDD-friendly)
- Clear interfaces between components
- Simplified extension to new GNSS systems or output formats
- Centralized error and warning handling

Refactor decisions were driven by the need to:
- Decouple parsing logic from computation and output
- Use Python dataclasses for structured data (ephemeris, observations, solutions)
- Enable dependency injection for configuration and logging, facilitating testability and mocking
- Support both CLI and library usage

#### Key Changes Made During the TDD Process

Development followed a scenario-driven, test-first approach:
- Each Gherkin scenario in `bdd-rtklib_gps_positioning.md` maps to a test module or case.
- Modules/classes expose methods designed for direct unit/integration testing.
- The TDD process led to the creation of clear error handling for malformed records, missing data, and unsupported observation types.
- The architecture evolved to support mixed satellite systems (e.g., GPS, GLONASS) and to gracefully degrade when encountering unsupported systems or data.

#### Current Structure and Responsibilities of Each Module

- **rinex.py**: 
  - `RinexHeaderParser`, `RinexDataParser`: Parse RINEX observation and navigation files.
  - `RinexObservation`, `RinexNavigation`: Data classes for parsed data.
- **ephemeris.py**: 
  - `EphemerisExtractor`: Extracts/manages satellite ephemeris.
  - `Ephemeris`: Data class for ephemeris parameters.
- **observation.py**: 
  - `ObservationExtractor`: Manages observation data (pseudorange, carrier phase, etc.).
  - `ObservationEpoch`: Data class for a single epoch’s observations.
- **pseudorange.py**: 
  - `PseudorangeCalculator`: Computes pseudoranges, applies corrections.
- **positioning.py**: 
  - `PositionSolver`: Calculates receiver position (SPP algorithm).
  - `PositionSolution`: Data class for position results and quality indicators.
- **system.py**: 
  - `SatelliteSystemManager`: Handles GNSS system logic (GPS, GLO, Galileo, BeiDou).
  - `SystemConfig`: Configures supported/active systems.
- **io.py**: 
  - `ResultFormatter`: Formats output (CSV, JSON).
  - `Logger`: Centralized logging for errors, warnings, info.
- **cli.py**: 
  - `main()`: CLI entry point.
  - `ArgumentParser`: Handles command-line arguments.

#### Important Notes for Future Contributors or Maintainers

- **Extensibility**: New GNSS systems can be added by extending `SatelliteSystemManager` and updating ephemeris/observation parsing logic.
- **Error Handling**: All modules should use the centralized `Logger` for reporting.
- **Testing**: Maintain the TDD approach—add Gherkin scenarios and corresponding tests for new features or bug fixes.
- **Output**: To add new output formats, extend `ResultFormatter`.
- **Configuration**: Use `SystemConfig` for enabling/disabling GNSS systems and for dependency injection in tests.
- **CLI/Library**: All logic should remain accessible both via CLI and as a Python library.

---
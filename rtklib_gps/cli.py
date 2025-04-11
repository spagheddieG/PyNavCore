"""
Command-line interface (CLI) module for RTKLIB-like GPS positioning script.

Functions:
    main(): Entry point for CLI usage.
"""

import argparse

def main():
    """Entry point for the CLI tool.

        Parses command-line arguments and orchestrates the processing pipeline.

        Raises:
            NotImplementedError: If the processing pipeline is not implemented.
        """
    parser = argparse.ArgumentParser(
        description="RTKLIB-like GPS Positioning Script (SPP, extensible to RTK)"
    )
    parser.add_argument(
        "--obs", required=True, help="Path to RINEX observation file"
    )
    parser.add_argument(
        "--nav", required=True, help="Path to RINEX navigation file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output file (CSV or JSON)"
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Output format"
    )
    parser.add_argument(
        "--systems", nargs="+", default=["GPS"], help="GNSS systems to use (e.g., GPS GLO GAL BDS)"
    )
    args = parser.parse_args()

    # Minimal implementation: indicate pipeline is not yet implemented
    raise NotImplementedError("Pipeline not yet implemented. CLI arguments parsed successfully.")

if __name__ == "__main__":
    main()
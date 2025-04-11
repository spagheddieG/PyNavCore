"""
I/O and logging module for RTKLIB-like GPS positioning script.

Classes:
    ResultFormatter: Formats output as CSV and JSON.
    Logger: Handles warnings, errors, and info messages.
"""

import json
import csv
from typing import List, Dict, Any, Optional

class ResultFormatter:
    """
    Formats position solutions and related data for output.

    Methods:
        to_csv(solutions: List[Any], file_path: str) -> None:
            Writes solutions to a CSV file.
        to_json(solutions: List[Any], file_path: Optional[str] = None) -> str:
            Returns solutions as a JSON string or writes to file if file_path is given.
    """
    def to_csv(self, solutions: List[Any], file_path: str) -> None:
        """Write solutions to a CSV file.

            Args:
                solutions (List[Any]): List of position solutions.
                file_path (str): Path to the output CSV file.

            Raises:
                NotImplementedError: If the method is not implemented.
            """
        raise NotImplementedError("CSV output logic not implemented.")

    def to_json(self, solutions: List[Any], file_path: Optional[str] = None) -> str:
        """Return solutions as a JSON string or write to file if file_path is given.

            Args:
                solutions (List[Any]): List of position solutions.
                file_path (Optional[str]): Path to the output JSON file.

            Returns:
                str: JSON string of solutions.

            Raises:
                NotImplementedError: If the method is not implemented.
            """
        raise NotImplementedError("JSON output logic not implemented.")

class Logger:
    """
    Centralized logger for warnings, errors, and info messages.

    Methods:
        info(msg: str) -> None
        warning(msg: str) -> None
        error(msg: str) -> None
    """
    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def warning(self, msg: str) -> None:
        print(f"[WARNING] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}")
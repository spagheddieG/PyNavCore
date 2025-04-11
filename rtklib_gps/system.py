"""
GNSS system abstraction and configuration module for RTKLIB-like GPS positioning script.

Classes:
    SatelliteSystemManager: Identifies and manages different GNSS systems (GPS, GLO, Galileo, BeiDou).
    SystemConfig: Configuration for supported/active systems.
"""

from typing import List, Dict, Any

class SatelliteSystemManager:
    """Identifies and manages GNSS systems and system-specific logic.

        Args:
            config (SystemConfig): Configuration for supported GNSS systems.
        """
    def __init__(self, config: 'SystemConfig'):
        self.config = config

    def identify_system(self, satellite_prn: str) -> str:
        """Return the GNSS system for a given satellite PRN.

            Args:
                satellite_prn (str): Satellite PRN identifier.

            Returns:
                str: GNSS system identifier.

            Raises:
                NotImplementedError: If the method is not implemented.
            """
        raise NotImplementedError("System identification logic not implemented.")

    def get_supported_systems(self) -> List[str]:
        """Return a list of supported GNSS systems.

            Returns:
                List[str]: Supported GNSS system identifiers.
            """
        return self.config.supported_systems

from dataclasses import dataclass, field
@dataclass
class SystemConfig:
    """Configuration for supported/active GNSS systems.

        Attributes:
            supported_systems (List[str]): Supported GNSS system identifiers (e.g., ['GPS', 'GLO']).
        """
    supported_systems: List[str] = field(default_factory=lambda: ['GPS'])
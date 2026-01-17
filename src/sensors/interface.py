from abc import ABC, abstractmethod
from ..core.types import SensorRecord

class SensorInterface(ABC):
    """
    Abstract base class for all sensors (Mock or Real).
    Section 9.1 in PRD.
    """
    
    @abstractmethod
    def read(self) -> SensorRecord:
        """
        Read the latest sensor data.
        """
        pass

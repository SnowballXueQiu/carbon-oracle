from pydantic import BaseModel, Field
from typing import Literal, Optional

class SensorRecord(BaseModel):
    """
    Standard sensor data record for a single timestamp.
    Section 4.2 in PRD.
    """
    time_min: int = Field(..., ge=0, description="Time in minutes")
    ph: float = Field(..., ge=0, le=14, description="pH value")
    conductivity: float = Field(..., ge=0, description="Conductivity in mS/cm")
    temperature: float = Field(..., description="Temperature in Celsius")
    color_index: float = Field(..., ge=0, le=1, description="Color index (0-1)")
    weight_change: float = Field(..., description="Weight change in grams")

class ExtractedFeatures(BaseModel):
    """
    Features extracted from a window of sensor records.
    Section 5.2 in PRD.
    """
    ph_final: float
    ph_slope: float
    temp_mean: float
    temp_std: float
    color_peak: float
    weight_loss: float

class PredictionResult(BaseModel):
    """
    Model prediction output.
    Section 6.2 in PRD.
    """
    capacity: float = Field(..., description="Predicted CO2 capacity")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")

class AgentDecision(BaseModel):
    """
    Decision made by the Agent.
    Section 7.3 in PRD.
    """
    action: Literal["continue", "warn", "stop"]
    reason: str
    adjustment: Optional[str] = None # e.g., "set_temp:750" 

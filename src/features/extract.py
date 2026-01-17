from typing import List
import numpy as np
from ..core.types import SensorRecord, ExtractedFeatures

class FeatureExtractor:
    """
    Extracts features from a sequence of sensor records.
    """
    
    def extract(self, records: List[SensorRecord]) -> ExtractedFeatures:
        if not records:
            # Return zero-filled features if no data
            return ExtractedFeatures(
                ph_final=0.0,
                ph_slope=0.0,
                temp_mean=0.0,
                temp_std=0.0,
                color_peak=0.0,
                weight_loss=0.0
            )

        # 1. ph_final (using latest)
        ph_final = records[-1].ph
        
        # 2. ph_slope (simple linear regression over window or just start-end)
        # Using simple (end - start) / time or similar. 
        # Better: (last - first) / count if count > 1
        if len(records) > 1:
            ph_slope = (records[-1].ph - records[0].ph) / (records[-1].time_min - records[0].time_min + 1e-6)
        else:
            ph_slope = 0.0

        # 3. temp_mean
        temps = [r.temperature for r in records]
        temp_mean = float(np.mean(temps))
        
        # 4. temp_std
        if len(records) > 1:
            temp_std = float(np.std(temps))
        else:
            temp_std = 0.0

        # 5. color_peak
        colors = [r.color_index for r in records]
        color_peak = max(colors)

        # 6. weight_loss (Total change so far)
        # weight_change is already "change", so we just take the last one relative to 0? 
        # PRD says "weight_change" field in record. 
        # If record.weight_change is cumulative (which mock generator implies: self.current_weight += ...), 
        # then we just take the last value.
        weight_loss = records[-1].weight_change

        return ExtractedFeatures(
            ph_final=ph_final,
            ph_slope=ph_slope,
            temp_mean=temp_mean,
            temp_std=temp_std,
            color_peak=color_peak,
            weight_loss=weight_loss
        )

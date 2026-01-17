from typing import List
from ..core.types import PredictionResult, ExtractedFeatures, AgentDecision
from ..core.config_loader import config

class AgentEngine:
    def __init__(self):
        self.min_capacity = config.get("min_thresholds.min_capacity", 1.5)
        self.target_capacity = config.get("min_thresholds.target_capacity", 3.0)
        self.require_confidence = config.get("agent.require_confidence", 0.6)
        
        self.history_predictions: List[float] = []

    def decide(self, prediction: PredictionResult, features: ExtractedFeatures, current_time: int = 0) -> AgentDecision:
        """
        Rule-based decision making.
        Section 7.4 in PRD.
        """
        # 0. Warmup Period Protection
        # Don't stop early batches that are still heating up (e.g. < 60 mins)
        if current_time < 60:
            return AgentDecision(
                action="continue",
                reason=f"Warmup Phase (T={current_time} < 60min). Monitoring..."
            )

        self.history_predictions.append(prediction.capacity)
        
        # 1. Check Confidence
        if prediction.confidence < self.require_confidence:
            return AgentDecision(
                action="warn",
                reason=f"Model confidence low ({prediction.confidence:.2f} < {self.require_confidence})"
            )
            
        # 1.5 CONTROL LOOP: Check for Over-heating (Simple Logic)
        # If temp > 850, we should cool down, regardless of prediction
        if features.temp_mean > 850:
            return AgentDecision(
                action="warn",
                reason=f"Temperature Critical: {features.temp_mean:.1f}C > 850C. Cooling down.",
                adjustment="set_temp:800"
            )

        # 2. Check Capacity Trend (Stop on Low Trend)
        # Trend is decreasing AND capacity < min_threshold
        is_decreasing = False
        if len(self.history_predictions) >= 3:
            # Simple check: last 3 are descending
            last_3 = self.history_predictions[-3:]
            if last_3[0] > last_3[1] > last_3[2]:
                is_decreasing = True
        
        if prediction.capacity < self.min_capacity and is_decreasing:
            return AgentDecision(
                action="stop",
                reason=f"Capacity {prediction.capacity:.2f} < {self.min_capacity} and trend is decreasing. Time to cut losses."
            )

        # 3. Check Success
        if prediction.capacity >= self.target_capacity:
            return AgentDecision(
                action="stop",
                reason=f"Target capacity {self.target_capacity} reached ({prediction.capacity:.2f}). Success!"
            )

        # 4. Default
        return AgentDecision(
            action="continue",
            reason="Process nominal. Monitoring..."
        )

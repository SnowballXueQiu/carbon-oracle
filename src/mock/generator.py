import random
import math
import numpy as np
from typing import Optional
from ..core.types import SensorRecord
from ..core.config_loader import config

class MockBatchGenerator:
    """
    Simulates a single experimental batch.
    """
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.duration = config.get("experiment_duration_min", 180)
        self.current_min = 0
        
        # Batch characteristics (Section 4.5)
        self.batch_type = self._determine_batch_type()
        
        # Initial States & Parameters
        self.start_ph = random.uniform(13.0, 14.0)
        self.target_temp = 800.0  # Default target
        
        # Modifiers based on batch type
        if self.batch_type == "under_active":
            self.target_temp = random.uniform(400, 600)
            self.ph_decay_rate = 0.5  # Slow decay
        elif self.batch_type == "over_active":
            self.target_temp = random.uniform(850, 950)
            self.ph_decay_rate = 1.5  # Fast decay
        elif self.batch_type == "abnormal":
            self.target_temp = 800
            self.ph_decay_rate = random.choice([0.1, 3.0]) 
        elif self.batch_type == "optimal":
            # Tuned for Success (Goal > 3.0 mmol/g)
            self.target_temp = 800.0 
            self.ph_decay_rate = 0.6 # Reduced from 1.2 to land near pH 8.0 (13.5 - 0.03*180 = ~8.1)
            self.start_ph = 13.5
        else: # Normal
            self.target_temp = random.uniform(750, 850)
            self.ph_decay_rate = 1.0

        # Dynamics (Initial State)
        self.current_ph = self.start_ph
        self.current_cond = 1.0
        self.current_temp = 25.0  # Room temp start
        self.current_color = 0.0
        self.current_weight = 0.0
        
        # History for calculating final capacity
        self.temp_history = []
        self.ph_history = []
        self.color_history = []

    def _determine_batch_type(self) -> str:
        # Increase probability of interesting/bad batches for Demo purposes
        r = random.random()
        if r < 0.60: return "optimal"       # 60% Optimal/Success Focus
        if r < 0.80: return "normal"        # 20% Normal (Variable)
        if r < 0.90: return "under_active"  # 10% Weak
        if r < 0.95: return "over_active"   # 5% Overcooked
        return "abnormal"                   # 5% Chaos

    def adjust_target_temp(self, new_temp: float):
        """ Control Interface for Agent """
        print(f"[MockHardware] Adjusting Heater Target: {self.target_temp:.1f} -> {new_temp:.1f}C")
        self.target_temp = new_temp

    def step(self) -> Optional[SensorRecord]:
        if self.current_min > self.duration:
            return None
        
        # Inject Chaos based on batch type
        chaos_factor = 1.0
        if self.batch_type == "abnormal": chaos_factor = 3.0
        if self.batch_type == "optimal": chaos_factor = 0.2 # Very stable
        
        # 1. Update Physics/Chemistry
        
        # pH: Decays exponentially or linearly
        decay = (0.05 * self.ph_decay_rate) 
        # Add random spikes
        noise = random.gauss(0, 0.05 * chaos_factor)
        self.current_ph = min(14.0, max(7.0, self.current_ph - decay + noise))
        
        # Temperature: PID-like approach to target
        target = self.target_temp
        # Abnormal: Temperature drifts or fails
        if self.batch_type == "abnormal" and self.current_min > 60:
             target = 400 # Heater failure simulation
             
        if self.current_temp < target:
            self.current_temp += (10.0 + random.gauss(0, 2.0 * chaos_factor))
        else:
            # Steady state fluctuation
            self.current_temp = target + random.gauss(0, 5.0 * chaos_factor)
            
        # Conductivity: Increases then stabilizes
        # Correlated with temperature and ion release
        sigmoid = 1 / (1 + math.exp(-(self.current_min - 30)/10)) 
        self.current_cond = 1.0 + (29.0 * sigmoid) + random.gauss(0, 0.5)

        # Color Index: 0 -> 1 as carbonization happens
        # Correlated with Time and Temp
        temp_factor = (self.current_temp / 800.0)
        self.current_color = min(1.0, self.current_color + (0.006 * temp_factor) + random.gauss(0, 0.01))
        self.current_color = max(0.0, self.current_color)

        # Weight Change: mostly loss
        weight_step = -0.003 * temp_factor 
        if self.current_min % 20 == 0: # Random adsorption event?
            weight_step += 0.01 
        self.current_weight = min(0.2, max(-0.5, self.current_weight + weight_step + random.gauss(0, 0.001)))

        # Update History
        self.temp_history.append(self.current_temp)
        self.ph_history.append(self.current_ph)
        self.color_history.append(self.current_color)

        record = SensorRecord(
            time_min=self.current_min,
            ph=round(self.current_ph, 2),
            conductivity=round(self.current_cond, 2),
            temperature=round(self.current_temp, 1),
            color_index=round(self.current_color, 3),
            weight_change=round(self.current_weight, 4)
        )
        
        self.current_min += 1
        return record

    def calculate_ground_truth_capacity(self) -> float:
        """
        Section 4.4: 
        co2_capacity = f(pH_final, temperature_mean, color_index_peak) + bias + noise
        """
        if not self.temp_history: return 0.0
        
        ph_final = self.ph_history[-1]
        temp_mean = np.mean(self.temp_history)
        color_index_peak = max(self.color_history)

        # Ideal conditions: pH ~ 8-9, Temp ~ 800, Color ~ 0.8
        
        # Score calculation
        score = 0.0
        
        # Temp Score: Bell curve around 800
        score += 2.0 * math.exp( -((temp_mean - 800)**2) / (2 * 50**2) )
        
        # pH Score: Lower is better (more activation?), but too low is bad? 
        # PRD: "pH drop too fast -> pore structure damaged". 
        # Actually usually activation implies pH change. Let's assume final pH around 8 is good.
        score += 1.0 * math.exp( -((ph_final - 8.0)**2) / (2 * 1.5**2) )

        # Color Score: Higher is better (carbonization)
        score += 0.5 * color_index_peak

        # Batch Bias
        bias = 0.0
        if self.batch_type == "under_active": bias = -1.0
        if self.batch_type == "over_active": bias = -0.5 # Overburn is bad too
        if self.batch_type == "abnormal": bias = -2.0
        if self.batch_type == "optimal": bias = 0.5 # Increased bias to ensure capacity > 3.0

        capacity = score + bias + random.gauss(0, 0.1)
        return max(0.1, round(capacity, 2))

class MockSensorSystem:
    def __init__(self):
        self.current_generator = None
        self.batch_count = 0

    def start_new_experiment(self):
        self.batch_count += 1
        batch_id = f"BATCH_{self.batch_count:03d}"
        self.current_generator = MockBatchGenerator(batch_id)
        return batch_id
    
    def adjust_control(self, cmd: str):
        if not self.current_generator: return
        if cmd.startswith("set_temp:"):
            try:
                val = float(cmd.split(":")[1])
                self.current_generator.adjust_target_temp(val)
            except:
                pass

    def read(self) -> Optional[SensorRecord]:
        if not self.current_generator:
            return None
        return self.current_generator.step()

    def get_ground_truth(self) -> float:
        if not self.current_generator: return 0.0
        return self.current_generator.calculate_ground_truth_capacity()

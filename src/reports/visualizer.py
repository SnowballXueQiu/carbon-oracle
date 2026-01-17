import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
from ..core.types import SensorRecord

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Set academic style
        sns.set_theme(style="whitegrid", context="paper")
        
    def generate_report_charts(self, batch_id: str, records: List[SensorRecord]):
        """
        Generates a composite visualization of the experiment.
        """
        if not records:
            return

        # Convert to DataFrame for easier plotting
        data = [r.model_dump() for r in records]
        df = pd.DataFrame(data)

        # 1. Main Process Variables (pH, Temp, Cond)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Batch {batch_id}: Carbon Activation Process Profile", fontsize=14, fontweight='bold')

        # pH Plot
        sns.lineplot(data=df, x='time_min', y='ph', ax=axes[0], color='tab:blue', linewidth=2)
        axes[0].set_ylabel('pH Level')
        axes[0].set_title('Acidity Evolution (pH)', loc='left', fontsize=10)
        axes[0].axhline(y=7.0, color='gray', linestyle='--', alpha=0.5)

        # Temperature Plot
        sns.lineplot(data=df, x='time_min', y='temperature', ax=axes[1], color='tab:red', linewidth=2)
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].set_title('Thermal Profile', loc='left', fontsize=10)
        
        # Conductivity Plot
        sns.lineplot(data=df, x='time_min', y='conductivity', ax=axes[2], color='tab:green', linewidth=2)
        axes[2].set_ylabel('Cond. (mS/cm)')
        axes[2].set_xlabel('Time (min)')
        axes[2].set_title('Conductivity & Ion Release', loc='left', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{batch_id}_process_chart.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Report] Saved process charts to {save_path}")

        # 2. Correlation Visual check (Color vs Weight) if needed
        # Just creating one main report chart for now.
        return save_path

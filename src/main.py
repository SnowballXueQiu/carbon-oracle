import time
import sys
import os
import logging
from datetime import datetime
from typing import List

# Rich Imports
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import SpinnerColumn, Progress, TextColumn
from rich.status import Status
from rich.markdown import Markdown
from rich.theme import Theme
from rich import box

# Ensure we can import modules if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config_loader import config
from src.core.types import SensorRecord
from src.mock.generator import MockSensorSystem
from src.features.extract import FeatureExtractor
from src.models.predictor import Predictor
from src.agent.engine import AgentEngine
from src.ai.provider import AIProviderFactory
from src.reports.analyst import BatchAnalyst
from src.core.database import ExperimentDatabase

# Initialize Console with Rainbow Markdown Theme
custom_theme = Theme({
    "markdown.h1": "bold red",
    "markdown.h2": "bold orange1",
    "markdown.h3": "bold yellow",
    "markdown.h4": "bold green",
    "markdown.h5": "bold cyan",
    "markdown.h6": "bold magenta",
    "markdown.table": "white",
    "markdown.strong": "bold white", 
})
console = Console(theme=custom_theme)

# Setup Logging
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "experiment_monitor.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CarbonOracle")

def create_header():
    return Panel(
        Text("Carbon Oracle | Intelligent Adsorbent Monitoring System v0.2", justify="center", style="bold cyan"),
        style="cyan",
        box=box.DOUBLE
    )

def create_status_table():
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Time (min)", justify="right", style="cyan", no_wrap=True)
    table.add_column("pH", justify="right", style="magenta")
    table.add_column("Temp (°C)", justify="right", style="red")
    table.add_column("Cond (mS/cm)", justify="right", style="green")
    table.add_column("Pred Cap (mmol/g)", justify="right", style="yellow")
    table.add_column("Conf", justify="right", style="blue")
    table.add_column("Agent Action", justify="center", style="bold")
    return table

def main():
    console.clear()
    console.print(create_header())
    
    # 1. Initialize System with Animation
    with console.status("[bold green]Initializing System Components...", spinner="dots") as status:
        time.sleep(1.0) # Fake delay for effect
        config.get("prediction_interval_min", 5) # access config to load it
        config_status = "[bold green]Config Loaded[/]"
        console.print(f"  ✓ {config_status}")
        
        prediction_interval = config.get("prediction_interval_min", 5)
        
        sensor_system = MockSensorSystem()
        extractor = FeatureExtractor()
        predictor = Predictor()
        agent = AgentEngine()
        
        # Database & Memory Loader
        db = ExperimentDatabase()
        count = db.get_total_experiments()
        console.print(f"  ✓ Database Connected: [bold]{count}[/] experiments detected")
        if count >= 5:
             # Retrain model if we have enough data
             console.print("  ✓ [bold magenta]Self-Correcting Model:[/] Retraining on persistent history...")
             predictor.fit_on_history(db)
        
        # AI Provider
        llm = None
        try:
            llm = AIProviderFactory.get_provider()
            console.print(f"  ✓ AI Provider: [bold blue]{config.get('ai_provider')}[/] connected")
            console.print(f"    (Model: [dim]{config.get('ollama.model')}[/])")
        except Exception as e:
            console.print(f"  [bold red]x Warning[/]: AI Provider failed ({e})")
        
        # Initialize Analyst
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        analyst = BatchAnalyst(ai_provider=llm, output_dir=reports_dir)
        console.print("  ✓ System Ready")

    # 2. Start Batch
    batch_id = sensor_system.start_new_experiment()
    # MOVED: get_ground_truth call to end of experiment
    
    console.print(Panel(f"Starting Experiment Batch: [bold yellow]{batch_id}[/]\nTarget: Adsorbent Activation", 
                       title="Experiment Control", border_style="green"))


    history_records: List[SensorRecord] = []
    
    # Live Dashboard Table
    table = create_status_table()

    # 3. Time Loop with Live Display
    active = True
    logger.info(f"Starting Batch {batch_id}")
    
    with Live(table, console=console, refresh_per_second=10) as live:
        while active:
            # Read Sensor
            record = sensor_system.read()
            if not record:
                console.print("[bold yellow]! Time Limit Reached[/]")
                logger.info("Time limit reached. Stopping.")
                break
                
            history_records.append(record)
            
            # Periodic Prediction
            if record.time_min % prediction_interval == 0:
                # Feature Eng
                feats = extractor.extract(history_records)
                
                # Predict
                prediction = predictor.predict(feats)
                
                # Agent Decide
                decision = agent.decide(prediction, feats)
                
                # Close-Loop Control Action
                control_msg = ""
                if decision.adjustment:
                    sensor_system.adjust_control(decision.adjustment)
                    control_msg = f" -> {decision.adjustment}"
                    logger.warning(f"Control Intervention: {decision.adjustment} (Reason: {decision.reason})")

                # Log Status
                logger.info(f"T={record.time_min} | pH={record.ph:.2f} | Temp={record.temperature:.1f} | "
                            f"Cap={prediction.capacity:.2f} | Action={decision.action}")

                # Style the Action
                action_style = "green"
                if decision.action == "warn": action_style = "yellow"
                if decision.action == "stop": action_style = "bold red"
                
                # Add Row to Table
                table.add_row(
                    str(record.time_min),
                    f"{record.ph:.2f}",
                    f"{record.temperature:.1f}",
                    f"{record.conductivity:.2f}",
                    f"{prediction.capacity:.2f}",
                    f"{prediction.confidence:.2f}",
                    f"[{action_style}]{decision.action.upper()}[/{action_style}]{control_msg}"
                )
                
                # Action Handling
                if decision.action == "stop":
                    live.console.print(Panel(f"[bold red]STOP SIGNAL TRIGGERED[/]\nReason: {decision.reason}", title="Agent Intervention", style="red"))
                    active = False
                elif decision.action == "warn":
                    pass

            # Simulation Speed
            time.sleep(0.02) 

    # 4. End of Experiment Report
    console.print("\n[bold cyan]Experiment Phase Completed.[/]")
    
    # Correctly retrieve Ground Truth AFTER experiment
    real_capacity_truth = sensor_system.get_ground_truth()

    if history_records:
        final_feats = extractor.extract(history_records)
        final_pred = predictor.predict(final_feats)
        
        # 1. Charts
        console.print("[bold cyan]Generating Visual Reports...[/]")
        analyst.visualizer.generate_report_charts(batch_id, history_records)
        console.print(f"[italic grey]Charts saved to: {analyst.visualizer.output_dir}[/]")

        # 2. AI Analysis (Streaming)
        if analyst.ai:
            # Prepare Prompt
            prompt = analyst.generate_report_prompt(batch_id, history_records, final_feats, final_pred, real_capacity_truth)
            
            console.print(Panel("[blink]Connecting to AI Agent...[/]", border_style="cyan"))
            
            full_response = ""
            with Live(Panel(Markdown(full_response), title="Real-time AI Assessment", border_style="cyan"), 
                      refresh_per_second=10, console=console) as live:
                try:
                    stream = analyst.ai.generate_stream(prompt)
                    for chunk in stream:
                        full_response += chunk
                        live.update(Panel(Markdown(full_response), title="Real-time AI Assessment", border_style="cyan"))
                except Exception as e:
                    full_response += f"\n**Analysis Error**: {e}"
                    live.update(Panel(Markdown(full_response), title="Real-time AI Assessment", border_style="red"))

            # Save to Memory
            analyst.save_analysis(batch_id, full_response, real_capacity_truth)
        else:
            console.print("[bold red]AI Provider not configured. Skipping qualitative analysis.[/]")

        # 5. Save to Knowledge Base (SQL DB for Training)
        db = ExperimentDatabase()
        db.save_experiment(batch_id, final_feats, real_capacity_truth, final_pred.capacity)
        console.print(f"[bold green]✓[/] Experiment Data saved to Persistent Memory.")

    else:
        console.print("[bold red]No data collected.[/]")

if __name__ == "__main__":
    main()

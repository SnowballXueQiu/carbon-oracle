# Carbon Oracle: A Multi-Modal "Soft-Sensing" Framework for Optimized Carbon Capture Material Synthesis

> **Research Prototype v0.2.1**  
> *Autonomous Experimentation System for Porous Carbon Activation*

---

## 1. Introduction and Objectives

**Carbon Oracle** is an experimental backend system designed to bridge the gap between bench-scale chemical synthesis and autonomous process control. 

The primary objective is to optimize the synthesis of **high-surface-area activated carbon** (for CO₂ capture applications) by solving the "Black Box" problem of high-temperature activation furnaces. Conventionally, chemists place precursors in a furnace for hours without visibility into the reaction trajectory. 

**Core Problem Solved:**
*   **Lack of In-Situ Visibility:** Interactions between activating agents (e.g., KOH, NaOH) and carbon precursors at 800°C are chaotic.
*   **Latency:** Quality verification (BET analysis) takes days after the experiment.
*   **Process Drift:** Identical parameters often yield different capacities due to precursor variability.

**Solution:**
Carbon Oracle implements a **Soft-Sensing** approach, allowing a digital agent to:
1.  **Monitor** proxy variables (pH of off-gas condensate, thermal fluctuations, conductivity) in real-time.
2.  **Predict** the final specific surface area (adsorption capacity) before the batch finishes.
3.  **Intervene** autonomously (adjusting temperature profiles) to salvage batches drifting towards failure.

---

## 2. Technical Methodology & Stack

The system employs a **Hybrid Neuro-Symbolic Architecture**, combining statistical machine learning for fast, quantitative predictions with Large Language Models (LLMs) for qualitative reasoning.

### Technical Stack
*   **Runtime**: Python 3.11+ (Managed via `uv`)
*   **Terminal UI**: `Textual` / `Rich` for high-frequency, low-latency visualization.
*   **Machine Learning**: `scikit-learn` (Random Forest Regressors) for capacity prediction.
*   **Vector Database**: `ChromaDB` for Retrieval Augmented Generation (RAG).
*   **Database**: `SQLite` for time-series sensor logging and training data persistence.
*   **LLM Integration**: Provider-agnostic interface (OpenAI / Ollama) for generating "Academic Batch Reports".

### key Innovations 
*   **Self-Correcting Predictor**: The ML model re-trains itself on startup using the accumulated history of "Real World" (database) experiments, allowing it to adapt to sensor drift over months.
*   **Memory-Augmented Analysis**: Before generating a batch report, the system queries a Vector DB for *semantically similar past experiments* (e.g., "Find other batches that overheated at T=60min"), enabling the AI to cite specific precedents.

---

## 3. Simulation Framework (Mock Data Generation)

To allow for rigorous testing of the control logic without wasting physical reagents, Carbon Oracle includes a high-fidelity **Phenomenological Simulator** (`src/mock/generator.py`).

The mock data is not random noise; it is generated based on theoretical chemical engineering principles governing activation kinetics.

### 3.1. Theoretical Basis (Phenomenological Modeling)
The simulator is grounded in the chemical kinetics of **KOH-assisted chemical activation** of carbon precursors. The core mechanism involves the redox reaction where carbon is etched by potassium compounds to create high-density microporosity.

*   **Reaction Kinetics & Temperature ($T$)**:
    *   **Mechanism**: The critical activation step is governed by the stoicheometry: $6KOH + 2C \longleftrightarrow 2K + 3H_2 + 2K_2CO_3$.
    *   **Thermodynamics**: Since the boiling point of Potassium is $762^\circ C$, effective activation requires operating at $T > 760^\circ C$ to generate metallic $K$ vapor, which intercalates between graphene layers to expand the lattice.
    *   **Dynamics**: Modeled as a PID response with thermal inertia: $T_{t+1} = T_{target} + \mathcal{N}(0, \sigma)$.
    
*   **pH Decay Dynamics ($\text{pH}$)**:
    *   **Chemical Significance**: The pH of the off-gas condensate serves as a real-time proxy for KOH consumption and reaction progress.
    *   **Trajectory**: The off-gas starts strongly alkaline ($\text{pH} \approx 14$) due to sublimated KOH. As the reaction proceeds and carbonate forms ($2KOH + CO_2 \to K_2CO_3 + H_2O$), the alkalinity naturally decays.
    *   **Model**: $\text{pH}_t = \text{pH}_{0} - k_{reaction} \cdot t + \epsilon$.
    *   **Process Window**: A rapid drop ($\text{pH} < 7$) indicates "Acid Runaway" (destructive decomposition), while sustained high pH implies insufficient precursor-activator contact.

*   **Conductivity & Gas Evolution ($\sigma$)**:
    *   **Ion Detection**: Peaks in conductivity correlate with the intense release of metallic $K$ vapor and hydrogen gas ($H_2$) during the peak activation window.
    *   **Model**: Modeled as a Sigmoid activation function centered around the activation onset time: $\sigma(t) = \frac{1}{1 + e^{-(t - t_{onset})}}$.

*   **Capacity Scoring Function ($C_{CO2}$)**:
    The final adsorption capacity (mmol/g) is estimated via a non-linear multivariate utility function representing the processing **"Goldilocks Zone"**:
    
    $Capacity \propto \alpha \cdot e^{-\frac{(T_{mean} - 800)^2}{2\sigma_T^2}} + \beta \cdot e^{-\frac{(pH_{final} - 8.0)^2}{2\sigma_{pH}^2}} + \gamma \cdot \text{Color}_{max}$

    *   **Efficiency**: This function rewards retention of micropores (formed optimally at $\approx 800^\circ C$) while penalizing sintering (caused by $T > 900^\circ C$) or incomplete activation (caused by low $T$).

### 3.2 Batch Scenarios
The simulator generates probabilistic scenarios covering the operational envelope:
1.  **Optimal (60%)**: Parameters stay within the calibrated window. Reference Capacity > 3.0 mmol/g.
2.  **Under-Active (10%)**: Temperature fails to reach threshold; Reaction incomplete.
3.  **Over-Active (5%)**: Thermal runaway; Micropores collapse into Macropores (Low surface area).
4.  **Abnormal/Chaos (5%)**: Sensors fail or erratic behavior.

---

## 4. Current Status & Roadmap

### ✅ Completed Features
*   [x] **Core Event Loop**: Synchronous sensor reading pipeline (10Hz).
*   [x] **TUI (Rainbow Mode)**: Real-time visualization of 4+ sensor streams and Agent decisions.
*   [x] **RAG Pipeline**: "Consulting" similar past experiments via Vector Search.
*   [x] **Streaming AI**: Character-level streaming of post-experiment analysis.
*   [x] **Auto-Training**: Model persistence and reloading.

### 🚧 Work in Progress (WIP)
*   **Hardware Interface Layer (`src/sensors/`)**: 
    *   Currently, the system interacts with `MockSensorSystem`. 
    *   **Next Step**: Implement `pyserial` / `minimalmodbus` drivers to read from:
        *   K-Type Thermocouples (via MAX6675/Arduino).
        *   Industrial pH probes (4-20mA loop).
    *   The architecture allows swapping the `Mock` class for a `Hardware` class without changing the main loop.
*   **Advanced Control**: Transitioning from Rule-Based (`if pH < 7: STOP`) to Reinforcement Learning (PPO) for temperature profiling.

---

## 5. Usage

### Prerequisites
*   Python 3.10+
*   `uv` package manager (recommended) or `pip`

### Installation
```bash
# Clone repository
git clone https://github.com/SnowballXueQiu/carbon-oracle.git
cd carbon-oracle

# Install dependencies (using uv)
uv sync
```

### Running the Demo
This launches the full simulation loop with TUI visualization.
```bash
./run_demo.sh
```

### Configuration
Edit `src/config/settings.yaml` to adjust:
*   `experiment_duration_min`: Length of simulation.
*   `ai_provider`: Toggle between `openai` and `ollama`.

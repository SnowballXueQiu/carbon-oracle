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

### 3.1. Theoretical Basis
The simulator models a standard **KOH Activation Process** of biomass.

*   **Temperature Dynamics ($T$)**:
    *   Modeled as a PID-controlled heating ramp with stochastic thermal lag.
    *   **Failure Mode**: "Abnormal" batches simulate thermocouple decoupling or heater element failure (drifting $\Delta T$).
    *   *Equation Logic*: $T_{t+1} = T_{target} + \mathcal{N}(0, \sigma_{chaos})$
    
*   **pH Decay ($\text{pH}$)**:
    *   **Proxy for Reaction Progress**: High pH indicates unreacted base. As K₂CO₃ forms and intercalates, free OH⁻ concentration drops.
    *   **Model**: Exponential decay with noise. 
    *   *Logic*: $\text{pH}(t) = \text{pH}_{0} - k \cdot t + \epsilon$
    *   **Optimal Trajectory**: A balanced decay (Rate $\approx 1.2$) correlates with optimal pore formation. Too fast = acid runaway; Too slow = insufficient activation.

*   **Conductivity ($\sigma$)**:
    *   Modeled as a Sigmoid function representing the release of metallic potassium vapor and ions during the "activation window" (>700°C).
    *   $S(t) = \frac{1}{1 + e^{-(t - t_0)}}$

*   **Ground Truth Capacity (Target Variable)**:
    The "True" CO₂ capacity (mmol/g) is calculated at the end of the batch using a multivariate scoring function representing the **"Goldilocks Zone"** of carbonization:
    $$ Capacity \propto f(T_{mean}) + f(pH_{final}) + f(Color_{max}) + \text{Bias}_{batch\_type} $$
    *   **$T_{mean}$**: Gaussian penalty centered at 800°C (Optimality).
    *   **$pH_{final}$**: Penalty for extreme acidity or alkalinity.
    
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

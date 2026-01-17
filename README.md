# Carbon Oracle: Intelligent Adsorbent Monitoring System
> **An AI-Driven Closed-Loop Control Framework for Carbon Capture Material Synthesis**

## Abstract
**Carbon Oracle** is an advanced "Soft Sensing" control system designed to optimize the synthesis of Reduced Graphene Oxide (rGO) based CO₂ adsorbents. By integrating real-time sensor fusion with a **Retrieval-Augmented Generation (RAG)** optimization engine, the system moves beyond passive monitoring to active, autonomous process intervention. The platform leverages a hybrid architecture combining random forest regressors for capacity prediction with a Vector Database (ChromaDB) for storing and retrieving experimental context, enabling a self-improving feedback loop that refines process parameters over successive batches.

---

## System Architecture

The system operates on four pillars of modern industrial autonomy:

### 1. **Soft-Sensing & Prediction (The "Oracle")**
*   **Feature Extraction**: Real-time extraction of temporal features (pH slope, dTemp/dt, colorimetric peaks) from raw sensor streams.
*   **Predictive Modeling**: Uses a self-correcting Random Forest regressor to estimate final CO₂ adsorption capacity (mmol/g) minutes into the reaction.

### 2. **Closed-Loop Control (The "Agent")**
*   **Autonomous Intervention**: A rule-based Agent Engine monitors safety and quality thresholds.
*   **Bi-Directional Action**: The system can actively adjust hardware parameters (e.g., specific heating rates, acid dosing) instead of merely alerting operators.

### 3. **Persistent Memory & Learning**
*   **Structured History (SQL)**: All sensor time-series and ground truth validations are committed to a `sqlite` database.
*   **Self-Training**: On startup, the `Predictor` module evaluates available historical data. If sufficient "Real World" ground truth exists, it automatically adapts its weights, evolving from synthetic priors to empirical reality.

### 4. **RAG-Enhanced Analysis**
*   **Vector Search (ChromaDB)**: Qualitative insights and experimental metadata are embedded and indexed.
*   **Contextual Intelligence**: Before generating a final report, the system queries the Knowledge Base for "nearest neighbor" experiments, providing the Large Language Model (LLM) with relevant historical precedents to ground its advice.

---

## Directory Structure

```
carbon-oracle/
├── src/
│   ├── agent/          # Control logic and intervention rules
│   ├── core/           # Database, Types, and Config loaders
│   ├── mock/           # High-fidelity chemical process simulator
│   ├── models/         # ML Predictors (Random Forest)
│   ├── reports/        # RAG-Analyst and Visualization engine
│   └── main.py         # Entry point and Event Loop
├── data/
│   ├── experiments.db  # Structured Experiment History (SQLite)
│   └── knowledge_base/ # Vector Embeddings (ChromaDB)
├── logs/               # Real-time process logs
├── reports/            # Generated charts and AI assessments
```

## Methodology

### The Optimization Loop
1.  **Process Start**: System initializes with target parameters.
2.  **In-Situ Monitoring**: Sensors sample at 1Hz; features are aggregated every `$prediction_interval`.
3.  **Inference**: Model predicts final capacity; Agent checks confidence bounds.
4.  **Intervention**: If `Temp > Critical` or `Trend == Negative`, Agent injects control signals (e.g., `set_temp:800`).
5.  **Post-Analysis**:
    *   Generates multi-modal report (Charts + Text).
    *   LLM reviews batch against similar historical cases (RAG).
    *   **Feedback**: LLM suggests optimized parameters for the *next* batch.

## Installation & Usage

### Prerequisites
*   Python 3.11+
*   `uv` package manager (recommended) or `pip`

### Setup
```bash
# Install dependencies
uv sync

# Run the Demo System
./run_demo.sh
```

### Configuration
Adjust thresholds and model settings in `src/config/config.toml`.

---
*© 2026 Carbon Oracle Research Group. All Rights Reserved.*

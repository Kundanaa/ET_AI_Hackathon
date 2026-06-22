# 🔋 Cell-to-Wheel Intelligence: EV Supply Chain & Asset Performance Management

An enterprise AI platform built for industrial EV ecosystems. This platform bridges the structural gap between upstream manufacturing quality management (QMS) and downstream operational fleet telematics (APM). By applying predictive machine learning and context-grounded Generative AI, it exposes hidden manufacturing defects before they trigger fleet failures or thermal runaway events.

## 🌟 Key Features
- **Predictive APM Engine:** Foresees Battery State of Health (SoH) and Remaining Useful Life (RUL) with an error margin of under 0.5%.
- **Upstream Defect Classification:** Correlates downstream charging thermal anomalies back to localized component batches to catch latent factory errors.
- **Grounded AI Operations Agent:** A conversational workspace powered by Gemini that interfaces directly with local CSV datasets for strict, data-only operational insights.

---

## 📊 Model Performance Metrics

### 1. Battery Degradation Regressor (APM)
- **RMSE:** `0.203% SoH`
- **R-Squared ($R^2$):** `0.932`
- **Impact:** Delivers predictive maintenance scheduling with near-perfect reliability, moving fleet ops away from reactive fixes.

### 2. Manufacturing Defect Classifier (QMS)
- **Accuracy:** `95.0%`
- **Precision:** `95.0%`
- **Recall:** `100.0%`
- **Feature Importances:** `Impurity_Level_PPM` (0.41), `Initial_Internal_Resistance` (0.40)
- **Impact:** 100% recall ensures no critical cell anomalies ever make it onto public or industrial haul routes.

---

## 🛠️ Project Structure & Setup

```text
├── data_generation.py     # Synthesizes the linked QMS & Telematics data
├── ml_pipeline.py         # Model training script for APM and QMS models
├── app.py                 # Interactive Streamlit Control Tower & AI Agent
├── requirements.txt       # Project python dependencies
└── datasets/
    ├── Fleet_Telematics_Data.csv
    └── QMS_Supply_Chain_Data.csv
```

## ⚙️ Installation & Running the Prototype
### Clone the repository:

```Bash

   git clone https://github.com/Kundanaa/ET_AI_Hackathon
   cd ev-cell-to-wheel-intelligence
```
### Install dependencies:

```Bash

   pip install -r requirements.txt
```

### Launch the Control Tower Dashboard: Configure your Google AI Studio API key inside app.py under the AI Agent configuration section, then boot up the interface:

```Bash

   streamlit run app.py
```

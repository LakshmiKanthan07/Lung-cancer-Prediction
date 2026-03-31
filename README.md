# 🫁 TeleMedLink | Oncology AI

**TeleMedLink** is an AI-powered lung cancer risk screening and data harmonization framework. It bridges the gap in discordant medical data by ingesting heterogeneous patient records, mapping them to a unified clinical ontology, and leveraging a highly calibrated XGBoost inference engine to predict cancer risk.

## ✨ Key Features
- **Data Harmonization Engine:** Ingests varying data formats (Clinical Codes, CDC Surveys, NLST study data) and normalizes them into a unified feature space.
- **Novel Feature Engineering:** Calculates advanced biological matrices including *Pack-Year Proxy*, *Biological Epistasis* (Gene-Environment Interactions), and *Toxic Accumulation*.
- **AI Inference (XGBoost):** An optimized, high-precision XGBoost classifier resilient to noise and tailored for medical diagnosis with custom scale-pos weights.
- **Interactive UI (Streamlit):**
  - **Patient Intake Form:** A live diagnostic tool with a three-tier risk banding system (Low, Elevated, Critical).
  - **Harmonization Logic Demo:** Visualizes the transformation of siloed raw data into actionable ML inputs.
  - **Ablation Study:** Explains the impact of the feature engineering layer with visual ROC and Feature Importance analysis.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telemedlink-oncology.git
   cd telemedlink-oncology
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit interface by running:
```bash
streamlit run app.py
```
*Note: Make sure the `final_harmonized_data.csv` is present in the root directory for the AI engine to initialize correctly.*

## 📂 Project Structure
- `app.py`: The main, fully-featured Streamlit application (telemedicine interface).
- `main.py`: Alternate/previous version of the Streamlit application.
- `compare.py`: Utility script for model comparison and initial data generation/validation.
- `final_harmonized_data.csv`: The processed training corpus.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Configured to ignore raw large datasets, environments, and cache files.

## ⚠️ Clinical Disclaimer
This tool is a research prototype to assist clinicians. It does **not** replace professional medical judgment. All outputs must be reviewed by a qualified physician.

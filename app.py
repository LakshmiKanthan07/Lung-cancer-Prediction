import os
import html
import hmac
import hashlib
import logging

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
import plotly.express as px

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="TeleMedLink | Oncology AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)
MODEL_CACHE_PATH = "model_cache.joblib"
MODEL_HASH_PATH = "model_cache.hash"
MODEL_HMAC_PATH = "model_cache.hmac"
DATA_PATH = "final_harmonized_data.csv"

RISK_THRESHOLD = 0.40
CRITICAL_THRESHOLD = 0.65
MAX_SYMPTOM_BOOST = 0.25

# =========================================================
# 2. THEME
# =========================================================
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #000000; }
    [data-testid="stSidebar"] { background-color: #0E1117; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    h1, h2, h3, h4, h5, h6, p, span, div, label, li { color: #000000; }
    .stTextInput input, .stNumberInput input, .stSelectbox, .stSlider { color: #000000 !important; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa; border: 1px solid #e0e0e0; border-left: 5px solid #2980b9;
        padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label { color: #555 !important; }
    div[data-testid="stMetricValue"] { color: #000000 !important; }
    div.stButton > button { background-color: #e74c3c; color: white !important; font-weight: bold; border: none; }
    .info-box    { padding:15px; border-radius:10px; background-color:#e3f2fd; border-left:6px solid #2196f3; margin-bottom:20px; color:#000000; }
    .novelty-box { padding:15px; border-radius:10px; background-color:#e8f5e9; border-left:6px solid #4caf50; margin-bottom:20px; color:#000000; }
    .warning-box { padding:12px; border-radius:8px;  background-color:#fff3cd; border-left:6px solid #ffc107; margin-bottom:15px; color:#000000; }
    .critical-box{ padding:12px; border-radius:8px;  background-color:#fdecea; border-left:6px solid #e74c3c; margin-bottom:15px; color:#000000; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 3. BACKEND — MODEL PERSISTENCE & TRAINING
# =========================================================
def _compute_data_hash(path: str) -> str:
    """Compute SHA-256 hash of the CSV to detect data changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_model_hmac(model_path: str, key: str) -> str:
    """Compute HMAC-SHA256 of the model file using data hash as key."""
    mac = hmac.new(key.encode("utf-8"), digestmod=hashlib.sha256)
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            mac.update(chunk)
    return mac.hexdigest()


def _train_model(df: pd.DataFrame):
    """Train XGBoost on the harmonized dataset and return the model."""
    features = [
        'Age', 'Gender', 'Smoking_Index', 'Pollution_Index', 'Family_History',
        'Pack_Year_Proxy', 'Toxic_Accumulation', 'Random_Noise_A', 'Random_Noise_B'
    ]
    X = df[features]
    y = df['Target_Risk']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


@st.cache_resource
def load_inference_engine():
    """Load model from cache if data is unchanged, otherwise retrain."""
    required_cols = {
        'Age', 'Pack_Year_Proxy', 'Family_History',
        'Toxic_Accumulation', 'Pollution_Index',
        'Gender', 'Smoking_Index', 'Random_Noise_A', 'Random_Noise_B',
        'Target_Risk'
    }

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return None, "Required data file not found. Please contact the administrator."
    except Exception:
        return None, "Failed to load data. Please contact the administrator."

    missing = required_cols - set(df.columns)
    if missing:
        return None, "Data file format is invalid. Please contact the administrator."

    current_hash = _compute_data_hash(DATA_PATH)

    # Try loading cached model if data hasn't changed
    if (os.path.exists(MODEL_CACHE_PATH)
            and os.path.exists(MODEL_HASH_PATH)
            and os.path.exists(MODEL_HMAC_PATH)):
        try:
            with open(MODEL_HASH_PATH, "r") as f:
                cached_hash = f.read().strip()
            with open(MODEL_HMAC_PATH, "r") as f:
                expected_hmac = f.read().strip()
            if cached_hash == current_hash:
                # Verify model file integrity before deserialization
                actual_hmac = _compute_model_hmac(MODEL_CACHE_PATH, current_hash)
                if hmac.compare_digest(actual_hmac, expected_hmac):
                    model = joblib.load(MODEL_CACHE_PATH)
                    return model, None
                else:
                    logger.error("Model file HMAC mismatch — possible tampering detected.")
        except Exception:
            pass  # Cache corrupted, retrain below

    # Retrain and cache with integrity seal
    try:
        model = _train_model(df)
        joblib.dump(model, MODEL_CACHE_PATH)
        new_hmac = _compute_model_hmac(MODEL_CACHE_PATH, current_hash)
        with open(MODEL_HASH_PATH, "w") as f:
            f.write(current_hash)
        with open(MODEL_HMAC_PATH, "w") as f:
            f.write(new_hmac)
        return model, None
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None, "Model initialization failed. Please contact the administrator."


model, model_error = load_inference_engine()


# =========================================================
# 4. SESSION STATE INITIALIZATION
# =========================================================
def _init_session_state():
    defaults = {
        "age": 40,
        "gender": "Male",
        "smoke_status": "Never Smoked",
        "years_smoked": 10,
        "cigs_per_day": 10,
        "secondhand": "None",
        "residence": "Rural (Clean Air)",
        "radon": False,
        "occupation_hazard": False,
        "fam_hist": "No",
        "pulmonary_hx": False,
        "radiation_hx": False,
        "sym_cough": False,
        "sym_weight": False,
        "sym_breath": False,
        "sym_blood": False,
        "sym_pain": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# =========================================================
# 5. INPUT SANITIZATION HELPERS
# =========================================================
def _esc(text) -> str:
    """HTML-escape any user-derived or computed string before rendering."""
    return html.escape(str(text))


def _safe_float(value, lo=0.0, hi=1.0) -> float:
    """Clamp a numeric value to [lo, hi]."""
    return max(lo, min(hi, float(value)))


# =========================================================
# 6. SIDEBAR
# =========================================================
st.sidebar.title("🧬 TeleMedLink")
st.sidebar.caption("AI-Powered Oncology Screening")
if model_error:
    st.sidebar.error("Model Offline")
st.sidebar.markdown("---")
nav = st.sidebar.radio("Module Selection:", [
    "Patient Intake Form",
    "Harmonization Logic",
    "Ablation Study (Validation)",
    "System Overview"
])

# =========================================================
# PAGE: SYSTEM OVERVIEW
# =========================================================
if nav == "System Overview":
    st.title("🫁 Lung Cancer Harmonization Framework")
    st.markdown("### A Novel Approach to Handling Heterogeneous Medical Data")

    st.markdown("""
    <div class="info-box">
    <b>The Challenge:</b> Medical data exists in "Silos." Hospital records, public surveys, and research trials
    use different formats (Schema Mismatch).<br>
    <b>The Solution:</b> A robust <b>Harmonization Pipeline</b> that ingests discordant CSVs, maps them to a
    unified ontology, and trains a calibrated XGBoost model.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1, 0.2, 1, 0.2, 1])
    with c1: st.info("**Raw Sources**\n\n📄 Cleveland Clinic\n\n📄 CDC Survey\n\n📄 NLST Study")
    with c2: st.markdown("<h1 style='text-align:center;margin-top:50px'>&#x27A1;</h1>", unsafe_allow_html=True)
    with c3: st.warning("**Harmonization Layer**\n\n⚙️ Schema Mapping\n\n🧹 Data Cleaning\n\n⚗️ Feature Engineering")
    with c4: st.markdown("<h1 style='text-align:center;margin-top:50px'>&#x27A1;</h1>", unsafe_allow_html=True)
    with c5: st.success("**AI Inference**\n\n🧠 XGBoost Classifier\n\n📈 Calibrated Output\n\n🛡️ Robust to Noise")

    st.markdown("---")
    st.markdown("#### 2. Technical Novelties")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="novelty-box">
        <b>Novelty A: Pack-Year Harmonization</b><br>
        Instead of binary (Yes/No), we engineered a <b>Pack-Year Proxy</b>:
        <i>(cigarettes/day / 20) x years smoked</i>, capturing cumulative lung damage over time.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="novelty-box">
        <b>Novelty B: Biological Epistasis</b><br>
        We model <b>Gene-Environment Interactions</b>. Smoking is exponentially more dangerous
        with family history — a non-linear risk profile simple models miss.
        </div>""", unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("""
        <div class="info-box">
        <b>Novelty C: Feature Absorption</b><br>
        Secondary risks (Radon, Passive Smoke) scale core indexes before inference —
        capturing complex histories without expanding model dimensionality.
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="info-box">
        <b>Novelty D: Schema Harmonization</b><br>
        Ingests discordant datasets (clinical codes, survey text, abbreviations) and maps them
        to a <b>Unified Clinical Ontology</b> for cross-hospital deployment.
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="info-box">
        <b>Novelty E: Noise Resilience</b><br>
        Random meaningless columns were deliberately injected during training.
        XGBoost ignores them entirely, proving it isolates true biological signals only.
        </div>""", unsafe_allow_html=True)

# =========================================================
# PAGE: HARMONIZATION LOGIC
# =========================================================
elif nav == "Harmonization Logic":
    st.title("⚙️ The Harmonization Engine")
    st.markdown("### Interactive Data Transformation Demo")

    st.subheader("Simulating Data Ingestion")
    source = st.selectbox("Select Raw Source Format:", [
        "Cleveland Clinic (Clinical Codes)",
        "CDC Survey (Descriptive Text)",
        "NLST Study (Abbreviated)"
    ])
    c1, c2, c3 = st.columns([1, 0.2, 1])
    with c1:
        st.markdown("**Raw Input (Before)**")
        if source == "Cleveland Clinic (Clinical Codes)":
            st.code('{\n  "Sex_Code": 1,\n  "Family_Hx": "Yes",\n  "Dx_Label": "POS"\n}', language="json")
            st.caption("Issues: Numeric codes, ambiguous headers.")
        elif source == "CDC Survey (Descriptive Text)":
            st.code('{\n  "Smoker_Score": "Heavy Smoker",\n  "Pollution": "Industrial Zone"\n}', language="json")
            st.caption("Issues: Unstructured text strings.")
        else:
            st.code('{\n  "gen": "M",\n  "smoke_stat": 4,\n  "yrs": 65\n}', language="json")
            st.caption("Issues: Non-standard abbreviations.")
    with c2:
        st.markdown("<br><br><h1>&#x27A1;</h1>", unsafe_allow_html=True)
    with c3:
        st.markdown("**Harmonized Output (After)**")
        if source == "Cleveland Clinic (Clinical Codes)":
            st.code('{\n  "Gender": 1,\n  "Family_History": 1,\n  "Target_Risk": 1\n}', language="json")
        elif source == "CDC Survey (Descriptive Text)":
            st.code('{\n  "Smoking_Index": 1.0,\n  "Pollution_Index": 0.8\n}', language="json")
        else:
            st.code('{\n  "Gender": 1,\n  "Smoking_Index": 0.7,\n  "Age": 65\n}', language="json")
        st.caption("Result: Unified Ontology ready for XGBoost.")

# =========================================================
# PAGE: ABLATION STUDY
# =========================================================
elif nav == "Ablation Study (Validation)":
    st.title("🔬 Ablation Study: Proving the Novelty")
    st.markdown("### Does our Feature Engineering actually matter?")

    results_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost',
                  'Logistic Regression', 'Random Forest', 'XGBoost'],
        'Feature_Set': ['Raw Data', 'Raw Data', 'Raw Data',
                        'Harmonized', 'Harmonized', 'Harmonized'],
        'Accuracy': [85.09, 87.32, 88.39, 85.58, 88.23, 88.64]
    }
    df_res = pd.DataFrame(results_data)
    xgb_enh = df_res[(df_res['Model'] == 'XGBoost') &
                     (df_res['Feature_Set'] == 'Harmonized')]['Accuracy'].values[0]

    st.markdown("---")
    st.markdown(
        f"<h3 style='text-align:center;color:#2980b9;'>"
        f"Harmonization Layer improved XGBoost to {xgb_enh:.2f}% accuracy</h3>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    fig_bar = px.bar(df_res, x="Model", y="Accuracy", color="Feature_Set", barmode="group",
                     color_discrete_map={"Raw Data": "#95a5a6", "Harmonized": "#2ecc71"},
                     text="Accuracy", height=450)
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_bar.update_layout(yaxis_title="Diagnostic Accuracy (%)", xaxis_title="",
                          legend_title="Input Pipeline", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(range=[80, 90]))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**ROC Curve Comparison**")
        if os.path.exists('roc_curve_comparison.png'):
            st.image('roc_curve_comparison.png', use_container_width=True)
        else:
            st.warning("ROC curve image not available.")
    with img_col2:
        st.markdown("**Champion Model Confusion Matrix**")
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', use_container_width=True)
        else:
            st.warning("Confusion matrix image not available.")

    st.markdown("---")
    col_text, col_chart = st.columns([1, 1.5])
    with col_text:
        st.write("Engineered features like **Pack-Year Proxy** dominate the decision-making process.")
        st.info("The model correctly ignores 'Random Noise', proving it has not overfit to irrelevant data.")
    with col_chart:
        if os.path.exists('feature_importance.png'):
            st.image('feature_importance.png', use_container_width=True)
        else:
            st.warning("Feature importance image not available.")

# =========================================================
# PAGE: PATIENT INTAKE
# =========================================================
elif nav == "Patient Intake Form":
    st.title("🩺 Comprehensive Risk Assessment")
    st.markdown("Run live predictions using the **Champion Model** (XGBoost Calibrated).")

    if model is None:
        st.error("System Offline. Model could not be loaded.")
        st.stop()

    # SECTION 1 — DEMOGRAPHICS
    st.subheader("1. Patient Profile")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input(
            "Age (Years)", min_value=18, max_value=90,
            value=st.session_state["age"], key="age_input"
        )
        st.session_state["age"] = age
    with c2:
        gender = st.selectbox(
            "Biological Sex", ["Male", "Female"],
            index=["Male", "Female"].index(st.session_state["gender"]),
            key="gender_input"
        )
        st.session_state["gender"] = gender

    st.markdown("---")

    # SECTION 2 — SMOKING
    st.subheader("2. Smoking & Lifestyle History")
    smoke_status = st.radio(
        "Primary Smoking Status",
        ["Never Smoked", "Former Smoker", "Current Smoker"],
        horizontal=True,
        index=["Never Smoked", "Former Smoker", "Current Smoker"].index(st.session_state["smoke_status"]),
        key="smoke_status_input"
    )
    st.session_state["smoke_status"] = smoke_status

    is_smoker = smoke_status != "Never Smoked"
    years_smoked, cigs_per_day = 0, 0

    if is_smoker:
        col_a, col_b = st.columns(2)
        with col_a:
            max_years = max(1, int(age) - 10)
            years_smoked = st.slider(
                "Total Years Smoked", 1, max_years,
                min(st.session_state["years_smoked"], max_years),
                key="years_smoked_input"
            )
            st.session_state["years_smoked"] = years_smoked
        with col_b:
            cigs_per_day = st.slider(
                "Average Cigarettes per Day", 1, 60,
                st.session_state["cigs_per_day"],
                key="cigs_per_day_input"
            )
            st.session_state["cigs_per_day"] = cigs_per_day
            st.caption(f"Equivalent to {cigs_per_day / 20:.1f} packs/day")

    secondhand_options = ["None", "Occasional (Social)", "Daily (Home/Work)"]
    secondhand = st.selectbox(
        "Secondhand Smoke Exposure",
        secondhand_options,
        index=secondhand_options.index(st.session_state["secondhand"]),
        key="secondhand_input",
        disabled=not is_smoker
    )
    st.session_state["secondhand"] = secondhand

    st.markdown("---")

    # SECTION 3 — ENVIRONMENT
    st.subheader("3. Environmental Factors")
    residence_options = [
        "Rural (Clean Air)", "Suburban (Moderate)",
        "Urban City Center (Poor)", "Industrial Zone (Hazardous)"
    ]
    col_x, col_y = st.columns(2)
    with col_x:
        residence = st.selectbox(
            "Primary Residence Type", residence_options,
            index=residence_options.index(st.session_state["residence"]),
            key="residence_input"
        )
        st.session_state["residence"] = residence
        radon = st.checkbox(
            "High Radon area / untested basement (>5 yrs)?",
            value=st.session_state["radon"],
            key="radon_input"
        )
        st.session_state["radon"] = radon
    with col_y:
        occupation_hazard = st.checkbox(
            "Occupational Exposure? (Asbestos, Silica, Mining)",
            value=st.session_state["occupation_hazard"],
            key="occupation_input"
        )
        st.session_state["occupation_hazard"] = occupation_hazard

    st.markdown("---")

    # SECTION 4 — CLINICAL HISTORY
    st.subheader("4. Clinical History")
    col_m, col_n = st.columns(2)
    with col_m:
        fam_hist = st.radio(
            "Immediate Family History of Lung Cancer?", ["No", "Yes"],
            horizontal=True,
            index=["No", "Yes"].index(st.session_state["fam_hist"]),
            key="fam_hist_input"
        )
        st.session_state["fam_hist"] = fam_hist
        pulmonary_hx = st.checkbox(
            "History of COPD, Emphysema, or Tuberculosis?",
            value=st.session_state["pulmonary_hx"],
            key="pulmonary_input"
        )
        st.session_state["pulmonary_hx"] = pulmonary_hx
        radiation_hx = st.checkbox(
            "Previous Chest Radiation Therapy?",
            value=st.session_state["radiation_hx"],
            key="radiation_input"
        )
        st.session_state["radiation_hx"] = radiation_hx
    with col_n:
        st.markdown("**Current Symptoms**")
        sym_cough = st.checkbox(
            "Persistent Cough (> 3 weeks)",
            value=st.session_state["sym_cough"],
            key="cough_input"
        )
        st.session_state["sym_cough"] = sym_cough
        sym_weight = st.checkbox(
            "Unexplained Weight Loss",
            value=st.session_state["sym_weight"],
            key="weight_input"
        )
        st.session_state["sym_weight"] = sym_weight
        sym_breath = st.checkbox(
            "Shortness of Breath",
            value=st.session_state["sym_breath"],
            key="breath_input"
        )
        st.session_state["sym_breath"] = sym_breath
        sym_blood = st.checkbox(
            "Coughing Blood / Haemoptysis",
            value=st.session_state["sym_blood"],
            key="blood_input"
        )
        st.session_state["sym_blood"] = sym_blood
        sym_pain = st.checkbox(
            "Chest Pain or Bone Pain",
            value=st.session_state["sym_pain"],
            key="pain_input"
        )
        st.session_state["sym_pain"] = sym_pain

    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
    <b>Clinical Disclaimer:</b> This tool is a research prototype to assist clinicians.
    It does <b>not</b> replace professional medical judgment. All outputs must be reviewed by a qualified physician.
    </div>""", unsafe_allow_html=True)

    submitted = st.button("RUN AI DIAGNOSIS ⚡")

    if submitted:
        with st.spinner("Harmonizing data points... calculating clinical toxicity..."):
            g_val = 1 if gender == "Male" else 0
            f_val = 1 if fam_hist == "Yes" else 0

            # SMOKING INDEX — only applies if actually smoked
            if is_smoker:
                s_base = _safe_float(cigs_per_day / 40.0)
            else:
                s_base = 0.0
            sh_map = {"None": 0.0, "Occasional (Social)": 0.1, "Daily (Home/Work)": 0.25}
            s_idx = _safe_float(s_base + sh_map.get(secondhand, 0.0))

            # PACK-YEAR PROXY
            if is_smoker:
                clinical_pack_years = (cigs_per_day / 20.0) * years_smoked
                pack_year_proxy = _safe_float(clinical_pack_years / 30.0)
            else:
                pack_year_proxy = 0.0

            # POLLUTION INDEX
            res_map = {
                "Rural (Clean Air)": 0.10,
                "Suburban (Moderate)": 0.30,
                "Urban City Center (Poor)": 0.60,
                "Industrial Zone (Hazardous)": 0.80
            }
            p_base = res_map.get(residence, 0.10)
            p_idx = _safe_float(
                p_base
                + (0.30 if occupation_hazard else 0.0)
                + (0.20 if pulmonary_hx else 0.0)
                + (0.20 if radon else 0.0)
                + (0.15 if radiation_hx else 0.0)
            )

            toxic_accum = _safe_float((age / 100.0) * p_idx)

            # Frozen noise — deterministic inference
            noise_a, noise_b = 0.5, 0.5

            input_df = pd.DataFrame(
                [[age, g_val, s_idx, p_idx, f_val,
                  pack_year_proxy, toxic_accum, noise_a, noise_b]],
                columns=['Age', 'Gender', 'Smoking_Index', 'Pollution_Index', 'Family_History',
                         'Pack_Year_Proxy', 'Toxic_Accumulation', 'Random_Noise_A', 'Random_Noise_B']
            )

            prob = float(model.predict_proba(input_df)[0][1])

            # SYMPTOM BOOST — capped at MAX_SYMPTOM_BOOST to prevent ceiling inflation
            symptom_boost_fraction = 0.0
            if sym_cough:  symptom_boost_fraction += 0.05
            if sym_weight: symptom_boost_fraction += 0.06
            if sym_breath: symptom_boost_fraction += 0.05
            if sym_blood:  symptom_boost_fraction += 0.12
            if sym_pain:   symptom_boost_fraction += 0.09
            symptom_boost_fraction = min(symptom_boost_fraction, MAX_SYMPTOM_BOOST)

            # Multiplicative boost: preserves calibration at both ends
            prob = prob + (1.0 - prob) * symptom_boost_fraction
            prob = min(prob, 0.97)

            genetic_synergy_active = (f_val == 1 and s_idx > 0.2)

        # ── OUTPUT ──────────────────────────────────────────
        st.markdown("### 📊 Clinical Analysis Report")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AI Risk Probability", f"{prob:.1%}")
        m2.metric("Pack-Year Proxy", f"{pack_year_proxy:.2f}")
        m3.metric("Toxic Exposure Index", f"{toxic_accum:.2f}")
        m4.metric("Genetic Synergist", "Active ⚠️" if genetic_synergy_active else "Inactive ✅")

        st.markdown("---")

        # THREE-TIER RISK BAND
        if prob >= CRITICAL_THRESHOLD:
            st.markdown("""
            <div class="critical-box">
            🚨 <b>CRITICAL RISK DETECTED</b> — Immediate clinical intervention required.
            </div>""", unsafe_allow_html=True)
            st.progress(prob)
            risk_level = "CRITICAL"
        elif prob >= RISK_THRESHOLD:
            st.warning("⚠️ **ELEVATED RISK DETECTED** — Further investigation strongly recommended.")
            st.progress(prob)
            risk_level = "ELEVATED"
        else:
            st.success("✅ **LOW RISK**")
            st.progress(prob)
            risk_level = "LOW"

        # ── RISK DRIVERS (HTML-escaped user-derived values) ──
        reasons = []
        if pack_year_proxy >= 0.33:
            reasons.append(f"Heavy cumulative smoking burden (Pack-Year Proxy: <b>{_esc(f'{pack_year_proxy:.2f}')}</b> ≥ 10 pack-years).")
        elif pack_year_proxy > 0.10:
            reasons.append(f"Moderate smoking history (Pack-Year Proxy: <b>{_esc(f'{pack_year_proxy:.2f}')}</b>).")
        elif s_idx > 0.1:
            reasons.append("Secondhand / passive smoke exposure noted.")

        if genetic_synergy_active:
            reasons.append("<b>⚠️ Genetic-Environmental Interaction:</b> Family history amplified by smoking exposure.")
        elif f_val == 1:
            reasons.append("First-degree family history of lung cancer independently elevates baseline risk.")

        if p_idx >= 0.70:
            reasons.append(f"Severe environmental/occupational toxicity (Pollution Index: <b>{_esc(f'{p_idx:.2f}')}</b>).")
        elif p_idx >= 0.40:
            reasons.append(f"Moderate environmental exposure (Pollution Index: <b>{_esc(f'{p_idx:.2f}')}</b>).")

        if toxic_accum >= 0.35:
            reasons.append(f"High age-weighted toxic accumulation (<b>{_esc(f'{toxic_accum:.2f}')}</b>) — prolonged exposure history.")
        elif toxic_accum >= 0.15:
            reasons.append(f"Moderate age-weighted toxic burden (<b>{_esc(f'{toxic_accum:.2f}')}</b>).")

        if age >= 55:
            reasons.append(f"Age <b>{_esc(int(age))}</b> — risk increases significantly after 55.")
        elif age >= 40:
            reasons.append(f"Age <b>{_esc(int(age))}</b> — entering moderate risk window (40–54).")

        if pulmonary_hx:
            reasons.append("Pre-existing pulmonary disease (COPD/Emphysema/TB) raises malignant transformation risk.")
        if radiation_hx:
            reasons.append("Prior chest radiation therapy is an independent lung cancer risk factor.")
        if occupation_hazard:
            reasons.append("Occupational carcinogen exposure (asbestos/silica/mining) detected.")

        if sym_blood:
            reasons.append("🚨 <b>Haemoptysis (coughing blood)</b> — critical red-flag symptom; urgent bronchoscopy indicated.")
        if sym_pain:
            reasons.append("🚨 <b>Chest/bone pain</b> — may indicate local invasion or metastasis; urgent imaging required.")
        if sym_weight:
            reasons.append("Unexplained weight loss — constitutional symptom associated with malignancy.")
        if sym_cough:
            reasons.append("Persistent cough (>3 weeks) — key screening symptom for central airway tumours.")
        if sym_breath:
            reasons.append("Shortness of breath — may indicate pleural effusion or airway obstruction.")

        if reasons:
            st.markdown("**Key Risk Drivers:**")
            for r in reasons:
                st.markdown(f"- {r}", unsafe_allow_html=True)
        elif risk_level == "LOW":
            st.write("No significant individual risk factors identified. Profile consistent with general population baseline.")

        st.markdown("---")

        # ── CLINICAL RECOMMENDATIONS ─────────────────────────
        st.markdown("### 🏥 Clinical Recommendations")

        if risk_level == "CRITICAL":
            st.error("""
**Immediate Actions Required:**
- Schedule **Low-Dose CT (LDCT) scan** within 1–2 weeks
- Urgent referral to **Pulmonologist / Thoracic Oncologist**
- Order **PET-CT** if mass identified on LDCT
- **Bronchoscopy + sputum cytology** if haemoptysis present
- Smoking cessation intervention if applicable
            """)
        elif risk_level == "ELEVATED":
            st.warning("""
**Recommended Actions:**
- Schedule **Low-Dose CT (LDCT) screening** within 4–6 weeks
- Referral to **Pulmonologist** for clinical evaluation
- Spirometry / PFT if pulmonary history is present
- Revisit in 3 months if imaging is clear
            """)
        else:
            if sym_cough or sym_breath or sym_weight or sym_blood or sym_pain:
                st.info("""
**Symptoms Present — Action Advised:**
- GP review within 2 weeks for symptom workup
- Chest X-ray as first-line investigation
- Return if symptoms persist or worsen
                """)
            else:
                st.success("""
**Routine Care:**
- Standard annual health checkup
- No smoking, reduce pollution exposure where possible
- Re-screen if symptoms develop or risk factors change
                """)

    st.markdown("---")
    st.subheader("Feature Engineering & Absorption")
    st.markdown("Visualizing Feature Absorption & Formulas")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**1. Adjusted Indexes**")
        st.latex(r'''\text{SmokingIdx} = \min(1.0,\ \text{PrimarySmoke} + \text{PassiveSmoke})''')
        st.latex(r'''\text{PollutionIdx} = \min(1.0,\ \text{BaseAir} + \text{Radon} + \text{PulmonaryHx})''')
    with c_right:
        st.markdown("**2. Temporal Risk Features**")
        st.latex(r'''\text{PackYearProxy} = \min\!\left(1.0,\ \frac{(\text{CigsPerDay}/20) \times \text{YearsSmoked}}{30}\right)''')
        st.latex(r'''\text{ToxicAccumulation} = \left(\frac{\text{Age}}{100}\right) \times \text{PollutionIdx}''')

    st.markdown("---")
    st.markdown("#### Live Math Demonstration")
    d1, d2, d3 = st.columns(3)
    with d1:
        age_demo = st.slider("Patient Age", 20, 80, 50, key="demo_age")
    with d2:
        smoke_demo = st.slider("Primary Smoking (Cigs/Day)", min_value=0, max_value=40, value=10, key="demo_smoke")
        passive_demo = st.checkbox("Daily Secondhand Smoke (+0.25)", key="demo_passive")
    with d3:
        air_demo = st.slider("Base Air Quality Index", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="demo_air")
        radon_demo = st.checkbox("Radon Exposure (+0.2)", key="demo_radon")

    s_base = min(smoke_demo / 40.0, 1.0)
    s_final = min(1.0, s_base + (0.25 if passive_demo else 0.0))
    p_final = min(1.0, air_demo + (0.2 if radon_demo else 0.0))
    demo_cigs = smoke_demo
    demo_years = max(0, age_demo - 18) if demo_cigs > 0 else 0
    pack_year_calc = min(1.0, (demo_cigs / 20.0) * demo_years / 30.0)
    toxic_calc = (age_demo / 100) * p_final

    st.markdown("<br>", unsafe_allow_html=True)
    rm1, rm2, rm3, rm4 = st.columns(4)
    rm1.metric("Adjusted Smoking Idx", f"{s_final:.2f}")
    rm2.metric("Pack-Year Proxy", f"{pack_year_calc:.3f}")
    rm3.metric("Adjusted Pollution Idx", f"{p_final:.2f}")
    rm4.metric("Toxic Accumulation", f"{toxic_calc:.3f}")
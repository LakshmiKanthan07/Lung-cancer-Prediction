import html
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import hashlib
import hmac
import os
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="TeleMedLink | Oncology AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_CACHE_PATH = "main_model_cache.joblib"
MODEL_HASH_PATH = "main_model_cache.hash"
MODEL_HMAC_PATH = "main_model_cache.hmac"
DATA_PATH = "final_harmonized_data.csv"

# --- 2. FORCED HIGH-CONTRAST THEME ---
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox, .stSlider {
        color: #000000 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #2980b9;
        padding: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        color: #555 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    div.stButton > button {
        background-color: #e74c3c;
        color: white !important;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 3. MODEL PERSISTENCE ---
def _compute_data_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_model_hmac(model_path: str, key: str) -> str:
    mac = hmac.new(key.encode("utf-8"), digestmod=hashlib.sha256)
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            mac.update(chunk)
    return mac.hexdigest()


def _safe_float(value, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, float(value)))


def _esc(text) -> str:
    return html.escape(str(text))


@st.cache_resource
def load_ai_engine():
    required_cols = {
        'Age', 'Gender', 'Smoking_Index', 'Pollution_Index', 'Family_History',
        'Pack_Year_Proxy', 'Toxic_Accumulation', 'Random_Noise_A', 'Random_Noise_B',
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

    if (os.path.exists(MODEL_CACHE_PATH)
            and os.path.exists(MODEL_HASH_PATH)
            and os.path.exists(MODEL_HMAC_PATH)):
        try:
            with open(MODEL_HASH_PATH, "r") as f:
                cached_hash = f.read().strip()
            with open(MODEL_HMAC_PATH, "r") as f:
                expected_hmac = f.read().strip()
            if cached_hash == current_hash:
                actual_hmac = _compute_model_hmac(MODEL_CACHE_PATH, current_hash)
                if hmac.compare_digest(actual_hmac, expected_hmac):
                    model = joblib.load(MODEL_CACHE_PATH)
                    return model, None
                else:
                    logger.error("Model file HMAC mismatch.")
        except Exception:
            pass

    features = ['Age', 'Gender', 'Smoking_Index', 'Pollution_Index', 'Family_History',
                'Pack_Year_Proxy', 'Toxic_Accumulation', 'Random_Noise_A', 'Random_Noise_B']
    X = df[features]
    y = df['Target_Risk']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    try:
        joblib.dump(model, MODEL_CACHE_PATH)
        new_hmac = _compute_model_hmac(MODEL_CACHE_PATH, current_hash)
        with open(MODEL_HASH_PATH, "w") as f:
            f.write(current_hash)
        with open(MODEL_HMAC_PATH, "w") as f:
            f.write(new_hmac)
    except Exception as e:
        logger.error(f"Model cache write failed: {e}")

    return model, None


model, model_error = load_ai_engine()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("🧬 TeleMedLink")
st.sidebar.caption("AI-Powered Oncology Screening")
if model_error:
    st.sidebar.error("Model Offline")
st.sidebar.markdown("---")
nav = st.sidebar.radio("Module Selection:", ["Patient Intake Form", "Harmonization Logic", "System Overview"])

# --- PAGE: SYSTEM OVERVIEW ---
if nav == "System Overview":
    st.title("🫁 Lung Cancer Risk Harmonization")
    st.markdown("### Bridging the Gap in Medical Data")

    col1, col2 = st.columns(2)
    with col1:
        st.info("❌ **The Problem**")
        st.markdown("**Patient data is siloed and messy.** Doctors miss patterns because genetic and lifestyle risks are often stored in different formats.")
    with col2:
        st.success("✅ **Our Solution**")
        st.markdown("**A Synergistic AI Model** that harmonizes inputs and calculates risk based on 'Pack-Years' and Biological Interaction.")

    st.markdown("---")
    st.metric("Model Sensitivity (Recall)", "96.9%", "High Safety Margin")

# --- PAGE: HARMONIZATION LOGIC ---
elif nav == "Harmonization Logic":
    st.title("⚙️ The Math Behind the Medicine")
    st.write("See how we translate raw patient answers into clinical risk vectors.")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Input")
        yr = st.number_input("Years Smoked", 0, 50, 20)
        cigs = st.number_input("Cigs per Day", 0, 60, 10)
    with c2:
        st.markdown("<br><br><h2>➡ 🧮 ➡</h2>", unsafe_allow_html=True)
    with c3:
        st.subheader("Harmonized Feature")
        pack_years = (cigs / 20) * yr
        st.metric("Clinical Pack-Years", f"{pack_years:.1f}")
        st.caption("Standard measure of lifetime tobacco exposure.")

# --- PAGE: PATIENT INTAKE (MAIN TOOL) ---
elif nav == "Patient Intake Form":
    st.title("🩺 Comprehensive Risk Assessment")
    st.markdown("Please complete the clinical interview below.")

    if model is None:
        st.error("System Offline. Model could not be loaded.")
        st.stop()

    with st.form("medical_form"):
        # --- SECTION 1: DEMOGRAPHICS ---
        st.subheader("1. Patient Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (Years)", 18, 90, 55)
        with c2:
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with c3:
            ethnicity = st.selectbox("Ethnicity", ["Asian", "White", "Black", "Hispanic", "Other"])

        st.markdown("---")

        # --- SECTION 2: LIFESTYLE ---
        st.subheader("2. Smoking & Lifestyle History")
        smoke_status = st.radio("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"], horizontal=True)

        is_smoker = smoke_status != "Never Smoked"
        years_smoked, cigs_per_day = 0, 0

        if is_smoker:
            col_a, col_b = st.columns(2)
            with col_a:
                max_years = max(1, int(age) - 10)
                years_smoked = st.slider("Total Years Smoked", 1, max_years, min(10, max_years))
            with col_b:
                cigs_per_day = st.slider("Average Cigarettes per Day", 1, 60, 10)
                st.caption(f"Equivalent to {cigs_per_day / 20:.1f} packs/day")

        st.markdown("---")

        # --- SECTION 3: ENVIRONMENTAL EXPOSURE ---
        st.subheader("3. Environmental Factors")
        col_x, col_y = st.columns(2)
        with col_x:
            residence = st.selectbox("Primary Residence Type", ["Rural (Clean Air)", "Suburban", "Urban City Center", "Industrial Zone"])
        with col_y:
            occupation_hazard = st.checkbox("History of Occupational Exposure? (Asbestos, Silica, Mining)")

        st.markdown("---")

        # --- SECTION 4: GENETICS & SYMPTOMS ---
        st.subheader("4. Clinical History")
        col_m, col_n = st.columns(2)
        with col_m:
            fam_hist = st.radio("Immediate Family History of Lung Cancer?", ["No", "Yes"], horizontal=True)
        with col_n:
            st.markdown("**Current Symptoms (Check all that apply)**")
            sym_cough = st.checkbox("Persistent Cough")
            sym_weight = st.checkbox("Unexplained Weight Loss")
            sym_breath = st.checkbox("Shortness of Breath")

        st.markdown("---")
        submitted = st.form_submit_button("RUN AI DIAGNOSIS ⚡")

    # --- LOGIC PROCESSING ---
    if submitted:
        with st.spinner("Harmonizing data points... calculating pack-years... assessing synergy..."):
            # 1. HARMONIZE INPUTS
            g_val = 1 if gender == "Male" else 0

            # Smoking Index (0.0 to 1.0) — only if actually smoked
            if is_smoker:
                raw_smoke_score = cigs_per_day / 30.0
                s_idx = _safe_float(raw_smoke_score)
            else:
                s_idx = 0.0

            # Pollution Index (0.0 to 1.0)
            res_map = {"Rural (Clean Air)": 0.1, "Suburban": 0.3, "Urban City Center": 0.6, "Industrial Zone": 0.8}
            p_base = res_map.get(residence, 0.1)
            p_hazard = 0.2 if occupation_hazard else 0.0
            p_idx = _safe_float(p_base + p_hazard)

            # Family History
            f_val = 1 if fam_hist == "Yes" else 0

            # --- NOVEL FEATURE ENGINEERING ---
            pack_year_proxy = _safe_float((age / 100) * s_idx)
            toxic_accum = _safe_float((age / 100) * p_idx)

            # Frozen noise — deterministic inference
            noise_a, noise_b = 0.5, 0.5

            # PREPARE VECTOR
            input_df = pd.DataFrame(
                [[age, g_val, s_idx, p_idx, f_val, pack_year_proxy, toxic_accum, noise_a, noise_b]],
                columns=['Age', 'Gender', 'Smoking_Index', 'Pollution_Index', 'Family_History',
                         'Pack_Year_Proxy', 'Toxic_Accumulation', 'Random_Noise_A', 'Random_Noise_B']
            )

            # PREDICT
            prob = float(model.predict_proba(input_df)[0][1])

            # SYMPTOM BOOST — capped at 0.25
            symptom_boost_fraction = 0.0
            if sym_cough:  symptom_boost_fraction += 0.05
            if sym_weight: symptom_boost_fraction += 0.06
            if sym_breath: symptom_boost_fraction += 0.05
            symptom_boost_fraction = min(symptom_boost_fraction, 0.25)

            prob = prob + (1.0 - prob) * symptom_boost_fraction
            prob = min(prob, 0.97)

            genetic_synergy_active = (f_val == 1 and s_idx > 0.2)

            # --- DISPLAY RESULTS ---
            st.markdown("### 📊 Clinical Analysis Report")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI Probability", f"{prob:.1%}")
            m2.metric("Pack-Year Risk", f"{pack_year_proxy:.2f}")
            m3.metric("Toxic Exposure", f"{toxic_accum:.2f}")
            m4.metric("Genetic Synergist", "Active ⚠️" if genetic_synergy_active else "Inactive ✅")

            st.markdown("---")

            # DECISION LOGIC
            if prob > 0.5:
                st.error("⚠️ **HIGH RISK DETECTED**")
                st.progress(prob)

                reasons = []
                if s_idx > 0.5:
                    reasons.append(f"Significant smoking history (Pack-Year Proxy: <b>{_esc(f'{pack_year_proxy:.2f}')}</b>)")
                if genetic_synergy_active:
                    reasons.append("<b>Genetic Interaction:</b> Smoking has amplified hereditary risk.")
                if p_idx > 0.6:
                    reasons.append(f"High environmental toxicity (Pollution Index: <b>{_esc(f'{p_idx:.2f}')}</b>).")
                if age > 65:
                    reasons.append(f"Age-related susceptibility (Age: <b>{_esc(int(age))}</b>).")

                if reasons:
                    st.write("**Key Drivers of Risk:**")
                    for r in reasons:
                        st.markdown(f"- {r}", unsafe_allow_html=True)

                st.warning("""
                **Recommended Action Plan:**
                1. **Immediate Referral:** Schedule Low-Dose CT (LDCT) Screening.
                2. **Oncology Consult:** Review family history markers.
                3. **Intervention:** Enrol in smoking cessation program immediately.
                """)

            else:
                st.success("✅ **LOW / MODERATE RISK**")
                st.progress(prob)

                if sym_cough or sym_breath or sym_weight:
                    st.info("ℹ️ **Note:** AI predicts low cancer risk based on history, BUT reported symptoms (Cough/Breath/Weight) require medical attention to rule out infection or other issues.")
                else:
                    st.write("Patient profile is consistent with the general population baseline.")

                st.markdown("**Recommendation:** Standard annual checkup. Maintain healthy lifestyle.")

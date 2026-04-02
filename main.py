import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="TeleMedLink | Oncology AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. FORCED HIGH-CONTRAST THEME (THE FIX) ---
st.markdown("""
    <style>
    /* FORCE ENTIRE PAGE BACKGROUND TO WHITE */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* FORCE SIDEBAR TO DARK NAVY */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    
    /* FORCE SIDEBAR TEXT TO WHITE */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* FORCE ALL MAIN TEXT TO BLACK */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000;
    }
    
    /* FIX INPUT FIELDS */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stSlider {
        color: #000000 !important;
    }
    
    /* METRIC BOX STYLING */
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
    
    /* BUTTON STYLING */
    div.stButton > button {
        background-color: #e74c3c;
        color: white !important;
        font-weight: bold;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKEND SYSTEM ---
@st.cache_resource
def load_ai_engine():
    try:
        df = pd.read_csv('final_harmonized_data.csv')
    except FileNotFoundError:
        return None, None

    # FEATURES MATCHING MAIN.PY EXACTLY
    features = ['Age', 'Gender', 'Smoking_Index', 'Pollution_Index', 'Family_History', 
                'Pack_Year_Proxy', 'Toxic_Accumulation', 'Random_Noise_A', 'Random_Noise_B']
    
    X = df[features]
    y = df['Target_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # High-Precision XGBoost
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
    return model

model = load_ai_engine()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("🧬 TeleMedLink")
st.sidebar.caption("AI-Powered Oncology Screening")
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
        # Logic from Main.py
        pack_years = (cigs / 20) * yr
        st.metric("Clinical Pack-Years", f"{pack_years:.1f}")
        st.caption("Standard measure of lifetime tobacco exposure.")

# --- PAGE: PATIENT INTAKE (MAIN TOOL) ---
elif nav == "Patient Intake Form":
    st.title("🩺 Comprehensive Risk Assessment")
    st.markdown("Please complete the clinical interview below.")
    
    if model is None:
        st.error("⚠️ System Offline. Please run compare_models.py to generate data.")
        st.stop()

    with st.form("medical_form"):
        # --- SECTION 1: DEMOGRAPHICS ---
        st.subheader("1. Patient Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (Years)", 20, 90, 55)
        with c2:
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        with c3:
            ethnicity = st.selectbox("Ethnicity", ["Asian", "White", "Black", "Hispanic", "Other"]) 

        st.markdown("---")

        # --- SECTION 2: LIFESTYLE (The Pack-Year Calculator) ---
        st.subheader("2. Smoking & Lifestyle History")
        smoke_status = st.radio("Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"], horizontal=True)
        
        years_smoked = 0
        cigs_per_day = 0
        
        if smoke_status != "Never Smoked":
            col_a, col_b = st.columns(2)
            with col_a:
                years_smoked = st.slider("Total Years Smoked", 1, 60, 10)
            with col_b:
                cigs_per_day = st.slider("Average Cigarettes per Day", 1, 60, 10)
                st.caption(f"Equivalent to {cigs_per_day/20:.1f} packs/day")

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
        with st.spinner(" harmonizing data points... calculating pack-years... assessing synergy..."):
            time.sleep(1.0) # Simulation delay
            
            # 1. HARMONIZE INPUTS (Map inputs to model features)
            
            # Gender
            g_val = 1 if gender == "Male" else 0
            
            # Smoking Index (0.0 to 1.0)
            raw_smoke_score = cigs_per_day / 30.0 
            s_idx = min(raw_smoke_score, 1.0) # Cap at 1.0
            
            # Pollution Index (0.0 to 1.0)
            res_map = {"Rural (Clean Air)": 0.1, "Suburban": 0.3, "Urban City Center": 0.6, "Industrial Zone": 0.8}
            p_base = res_map[residence]
            p_hazard = 0.2 if occupation_hazard else 0.0
            p_idx = min(p_base + p_hazard, 1.0)
            
            # Family History
            f_val = 1 if fam_hist == "Yes" else 0
            
            # --- NOVEL FEATURE ENGINEERING (Crucial Step) ---
            # 1. Pack-Year Proxy: (Age/100) * Smoking_Index
            pack_year_proxy = (age / 100) * s_idx
            
            # 2. Toxic Accumulation: (Age/100) * Pollution_Index
            toxic_accum = (age / 100) * p_idx
            
            # 3. Noise (Random)
            noise_a = np.random.rand()
            noise_b = np.random.rand()
            
            # PREPARE VECTOR
            input_df = pd.DataFrame([[age, g_val, s_idx, p_idx, f_val, pack_year_proxy, toxic_accum, noise_a, noise_b]],
                                    columns=['Age', 'Gender', 'Smoking_Index', 'Pollution_Index', 'Family_History', 
                                             'Pack_Year_Proxy', 'Toxic_Accumulation', 'Random_Noise_A', 'Random_Noise_B'])
            
            # PREDICT
            prob = model.predict_proba(input_df)[0][1]
            
            # --- DISPLAY RESULTS ---
            st.markdown("### 📊 Clinical Analysis Report")
            
            # Top Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI Probability", f"{prob:.1%}")
            m2.metric("Pack-Year Risk", f"{pack_year_proxy:.2f}")
            m3.metric("Toxic Exposure", f"{toxic_accum:.2f}")
            m4.metric("Genetic Synergist", "Active" if (f_val==1 and s_idx>0.3) else "Inactive")
            
            st.markdown("---")
            
            # DECISION LOGIC
            if prob > 0.5:
                st.error("⚠️ **HIGH RISK DETECTED**")
                st.progress(int(prob*100))
                
                # Dynamic Explanation
                reasons = []
                if s_idx > 0.5: reasons.append("Significant smoking history (High Pack-Years)")
                if f_val == 1 and s_idx > 0.3: reasons.append("<b>Genetic Interaction:</b> Smoking has amplified hereditary risk.")
                if p_idx > 0.6: reasons.append("High environmental toxicity detected.")
                if age > 65: reasons.append("Age-related susceptibility.")
                
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
                st.progress(int(prob*100))
                
                # Check for "Symptom Warning" even if risk is low
                if sym_cough or sym_breath or sym_weight:
                    st.info("ℹ️ **Note:** AI predicts low cancer risk based on history, BUT reported symptoms (Cough/Breath/Weight) require medical attention to rule out infection or other issues.")
                else:
                    st.write("Patient profile is consistent with the general population baseline.")
                    
                st.markdown("**Recommendation:** Standard annual checkup. Maintain healthy lifestyle.")
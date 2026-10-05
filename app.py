"""
Streamlit Application: Intelligent Dementia Detection System
Interactive Clinical Decision Support Web Application (3-Class Staging & Decision Tree Pipeline)
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

from database.db import (
    init_db,
    add_patient,
    add_prediction,
    get_all_records,
    delete_patient
)

# Configure professional wide page layout
st.set_page_config(
    page_title="Intelligent Dementia Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database safely at startup
try:
    init_db()
except Exception as e:
    st.error(f"Database initialization error: {e}")

# Professional Clinical CSS
st.markdown("""
<style>
    /* Global Typography & Palette */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1E293B;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E293B;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.4rem;
        margin-top: 1.2rem;
        margin-bottom: 1.0rem;
    }
    .category-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 0.75rem;
    }
    
    /* Clinical Assessment Cards */
    .card-normal {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 5px solid #16A34A;
        border-radius: 8px;
        padding: 1.25rem;
        color: #14532D;
    }
    .card-mild {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        border-radius: 8px;
        padding: 1.25rem;
        color: #78350F;
    }
    .card-dementia {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        border-radius: 8px;
        padding: 1.25rem;
        color: #7F1D1D;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .badge-mild {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .badge-dementia {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BENCHMARK PATIENT PROFILES (OASIS Dataset Reference Standards)
# -----------------------------------------------------------------------------
BENCHMARK_PROFILES = {
    "Manual Input (Custom Patient)": {
        "gender": "Female",
        "age": 74.0,
        "educ_level": 2, # Level 2: High School
        "ses": 3.0,
        "mmse": 29.0,
        "etiv": 1344.0,
        "nwbv": 0.743,
        "asf": 1.306,
        "cdr": None,
        "ground_truth": None,
        "id": "Custom Patient"
    },
    "Patient 1 (OAS1_0001) - Non-Demented Reference": {
        "gender": "Female",
        "age": 74.0,
        "educ_level": 2, # High School (12 yrs)
        "ses": 3.0,
        "mmse": 29.0,
        "etiv": 1344.0,
        "nwbv": 0.743,
        "asf": 1.306,
        "cdr": 0.0,
        "ground_truth": "Cognitively Normal (CDR 0.0)",
        "id": "OAS1_0001"
    },
    "Patient 2 (OAS1_0003) - Very Mild Stage Reference": {
        "gender": "Female",
        "age": 73.0,
        "educ_level": 4, # College Graduate (16 yrs)
        "ses": 3.0,
        "mmse": 27.0,
        "etiv": 1454.0,
        "nwbv": 0.708,
        "asf": 1.207,
        "cdr": 0.5,
        "ground_truth": "Very Mild Impairment (CDR 0.5)",
        "id": "OAS1_0003"
    },
    "Patient 3 (OAS1_0028) - Clinical Dementia Reference": {
        "gender": "Male",
        "age": 74.0,
        "educ_level": 5, # Post-Graduate (18 yrs)
        "ses": 1.0,
        "mmse": 21.0,
        "etiv": 1588.0,
        "nwbv": 0.681,
        "asf": 1.108,
        "cdr": 1.0,
        "ground_truth": "Clinical Dementia (CDR 1.0)",
        "id": "OAS1_0028"
    }
}

EDUC_OPTIONS = {
    1: "Level 1: < High School (< 12 years)",
    2: "Level 2: High School (12-13 years)",
    3: "Level 3: Some College (14-15 years)",
    4: "Level 4: College Graduate (16-17 years)",
    5: "Level 5: Post-Graduate (18+ years)"
}

EDUC_TO_YEARS = {1: 8.0, 2: 12.0, 3: 14.0, 4: 16.0, 5: 18.0}

# -----------------------------------------------------------------------------
# 3-CLASS MODEL TRAINING & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
def preprocess_dataframe_features(df, feature_cols):
    df_out = df.copy()
    if 'M/F' in df_out.columns:
        df_out['M/F'] = df_out['M/F'].apply(
            lambda x: 1.0 if str(x).strip().upper() in ['M', 'MALE', '1', '1.0'] else 0.0
        ).astype(float)
    for col in feature_cols:
        if col != 'M/F' and col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
            med = df_out[col].median()
            df_out[col] = df_out[col].fillna(med if pd.notna(med) else 0.0).astype(float)
    return df_out

@st.cache_resource
def get_trained_3class_pipeline():
    data_path = "dementia_dataset_2.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "cleaned_oasis.csv")
        
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['CDR']).copy()
    
    def map_3class(cdr):
        try:
            c = float(cdr)
            if c == 0.0:
                return 0
            elif c == 0.5:
                return 1
            else:
                return 2
        except:
            return 0
            
    df['target_3class'] = df['CDR'].apply(map_3class)
    feature_cols = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    df_clean = preprocess_dataframe_features(df, feature_cols)
    
    X = df_clean[feature_cols]
    y = df['target_3class']
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_split=3,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_scaled, y)
    
    return model, scaler, feature_cols

# Initialize model
model_3class, scaler_3class, feature_cols_3class = get_trained_3class_pipeline()

# =============================================================================
# SIDEBAR: BENCHMARK PROFILES & CLINICAL REFERENCE
# =============================================================================
with st.sidebar:
    st.subheader("Clinical Cohort Profiles")
    st.caption("Select a validated OASIS reference profile or input custom clinical parameters.")
    
    selected_benchmark = st.selectbox(
        "Patient Profile Selector:",
        list(BENCHMARK_PROFILES.keys()),
        index=0
    )
    
    profile = BENCHMARK_PROFILES[selected_benchmark]
    
    if profile["cdr"] is not None:
        st.markdown(f"""
        **Subject ID:** `{profile['id']}`  
        **Reference Stage:** {profile['ground_truth']}  
        **Ground-Truth CDR:** `{profile['cdr']}`
        """)
    else:
        st.caption("Manual patient entry active.")

    st.markdown("---")
    st.markdown("##### Clinical Staging Criteria")
    st.markdown("""
    - **CDR 0.0**: Normal Cognitive Function
    - **CDR 0.5**: Very Mild Cognitive Impairment / Prodromal
    - **CDR 1.0+**: Established Clinical Dementia
    """)

# =============================================================================
# MAIN INTERFACE: HEADER & TABS
# =============================================================================
st.markdown("<div class='main-title'>Intelligent Dementia Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Clinical Decision Support System for Multi-Class Dementia Staging & Risk Stratification based on OASIS Benchmarks.</div>", unsafe_allow_html=True)

# Application Navigation Tabs
tab_predict, tab_records = st.tabs(["Clinical Staging Assessment", "Patient Records Database"])

with tab_predict:
    with st.expander("System Architecture & Processing Workflow", expanded=False):
        st.markdown("""
        ```mermaid
        graph TD
            A[Patient Input / Benchmark Profiles] --> B[Pipeline Preprocessing & Encoding]
            B --> C[ML Model Inference]
            C --> D[Clinical Staging & Diagnostic Card]
            C --> E[Class Probability Distribution]
            C --> F[Parameter Summary & OASIS Validation]
        ```
        """)
        st.markdown("""
        **Pipeline Architecture:**
        1. **Clinical Inputs (8 Features)**: Demographic, cognitive, and volumetric anatomical indices.
        2. **Preprocessing & Encoding**: Categorical normalization, median imputation, and RobustScaler transformation.
        3. **ML Inference Pipeline**: Balanced multi-class Decision Tree / Ensemble Classifier.
        4. **Diagnostic Staging**: 3-Class clinical staging into *No Dementia*, *Very Mild Impairment*, or *Clinical Dementia*.
        """)

    st.markdown("<div class='section-header'>1. Interactive Clinical Parameter Input Form</div>", unsafe_allow_html=True)

    with st.form("clinical_input_form"):
        col_demo, col_neuro = st.columns(2)
        
        with col_demo:
            st.markdown("<div class='category-header'>Demographic & Socioeconomic Factors</div>", unsafe_allow_html=True)
            
            gender_default_idx = 0 if profile["gender"] == "Female" else 1
            gender_input = st.selectbox(
                "Biological Sex (M/F)",
                options=["Female", "Male"],
                index=gender_default_idx,
                help="Female is internally encoded as 0, Male as 1."
            )
            
            age_input = st.number_input(
                "Patient Age (years)",
                min_value=18.0,
                max_value=105.0,
                value=float(profile["age"]),
                step=1.0,
                help="Patient age in years (18 to 105)."
            )
            
            educ_level_input = st.selectbox(
                "Education Level (Educ)",
                options=[1, 2, 3, 4, 5],
                index=profile["educ_level"] - 1,
                format_func=lambda x: EDUC_OPTIONS[x],
                help="5-tier ordinal education scale."
            )
            
            ses_input = st.selectbox(
                "Socioeconomic Status (SES)",
                options=[1.0, 2.0, 3.0, 4.0, 5.0],
                index=int(profile["ses"]) - 1,
                help="Hollingshead Index (Class 1: Highest Status to Class 5: Lowest Status)."
            )
            
        with col_neuro:
            st.markdown("<div class='category-header'>Cognitive & Neuroanatomical Volumetric Indicators</div>", unsafe_allow_html=True)
            
            mmse_input = st.number_input(
                "MMSE Score (0 - 30)",
                min_value=0.0,
                max_value=30.0,
                value=float(profile["mmse"]),
                step=1.0,
                help="Mini-Mental State Examination score (0 to 30; standardized screening)."
            )
            
            etiv_input = st.number_input(
                "Estimated Total Intracranial Volume (eTIV, mm³)",
                min_value=800.0,
                max_value=2500.0,
                value=float(profile["etiv"]),
                step=1.0,
                help="Total cranial cavity volume in mm³ (800 - 2500)."
            )
            
            nwbv_input = st.number_input(
                "Normalized Whole Brain Volume (nWBV)",
                min_value=0.500,
                max_value=0.950,
                value=float(profile["nwbv"]),
                step=0.001,
                format="%.3f",
                help="Brain tissue volume fraction relative to eTIV (0.500 - 0.950)."
            )
            
            asf_input = st.number_input(
                "Atlas Scaling Factor (ASF)",
                min_value=0.700,
                max_value=1.800,
                value=float(profile["asf"]),
                step=0.001,
                format="%.3f",
                help="Geometric scaling factor normalizing brain dimensions to atlas space (0.700 - 1.800)."
            )
            
        submit_btn = st.form_submit_button("Run Clinical Prediction", use_container_width=True)

    # -------------------------------------------------------------------------
    # Inference & Diagnostic Output
    # -------------------------------------------------------------------------
    gender_code = 1 if gender_input == "Male" else 0
    educ_years = EDUC_TO_YEARS[educ_level_input]

    input_features = {
        'M/F': gender_code,
        'Age': age_input,
        'EDUC': educ_years,
        'SES': ses_input,
        'MMSE': mmse_input,
        'eTIV': etiv_input,
        'nWBV': nwbv_input,
        'ASF': asf_input
    }

    input_df = pd.DataFrame([input_features])[feature_cols_3class]
    input_scaled = scaler_3class.transform(input_df)

    predicted_class = model_3class.predict(input_scaled)[0]
    probabilities = model_3class.predict_proba(input_scaled)[0]

    prob_no_dementia = probabilities[0]
    prob_very_mild = probabilities[1]
    prob_dementia = probabilities[2]

    STAGES = {
        0: {
            "title": "No Dementia (Cognitively Normal)",
            "card_class": "card-normal",
            "badge": "STAGE: NORMAL (CDR 0.0)",
            "badge_class": "badge-normal",
            "description": "Patient exhibits normal cognitive and volumetric indicators consistent with healthy aging. No clinical dementia detected."
        },
        1: {
            "title": "Very Mild Dementia (Prodromal Stage)",
            "card_class": "card-mild",
            "badge": "STAGE: VERY MILD (CDR 0.5)",
            "badge_class": "badge-mild",
            "description": "Clinical indicators suggest early-stage cognitive impairment or prodromal dementia. Routine longitudinal monitoring and clinical assessment recommended."
        },
        2: {
            "title": "Clinical Dementia (Established Stage)",
            "card_class": "card-dementia",
            "badge": "STAGE: DEMENTIA (CDR 1.0+)",
            "badge_class": "badge-dementia",
            "description": "Neuroanatomical volumetric parameters and cognitive scores indicate established clinical dementia. Clinical specialist evaluation recommended."
        }
    }

    stage_info = STAGES[predicted_class]

    # Save to SQLite database upon submission
    if submit_btn:
        try:
            patient_record = {
                "patient_name": str(profile.get("id", "Custom Patient")),
                "gender": gender_input,
                "age": float(age_input),
                "educ_level": int(educ_level_input),
                "ses": float(ses_input),
                "mmse": float(mmse_input),
                "etiv": float(etiv_input),
                "nwbv": float(nwbv_input),
                "asf": float(asf_input)
            }
            new_patient_id = add_patient(patient_record)
            add_prediction(
                patient_id=new_patient_id,
                prediction_label=stage_info["title"],
                probability=float(probabilities[predicted_class]),
                risk_level=stage_info["badge"],
                model_name="RandomForest 3-Class Pipeline"
            )
            st.success("Record saved")
        except Exception as db_err:
            st.error(f"Database error while saving record: {db_err}")

    st.markdown("<div class='section-header'>2. Diagnostic Staging & Probability Distribution</div>", unsafe_allow_html=True)

    col_card, col_prob = st.columns([1.2, 1.0])

    with col_card:
        st.markdown(f"""
        <div class='{stage_info["card_class"]}'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                <h4 style='margin: 0; font-size: 1.25rem; font-weight: 700;'>{stage_info["title"]}</h4>
                <span class='status-badge {stage_info["badge_class"]}'>{stage_info["badge"]}</span>
            </div>
            <p style='font-size: 0.95rem; margin-bottom: 0.75rem; line-height: 1.4;'>{stage_info["description"]}</p>
            <div style='font-size: 0.9rem; font-weight: 600; border-top: 1px solid rgba(0,0,0,0.08); padding-top: 0.5rem;'>
                Model Diagnostic Confidence: <span style='font-size: 1.05rem;'>{probabilities[predicted_class]:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        st.markdown("<div class='category-header'>Classification Probability Distribution</div>", unsafe_allow_html=True)
        
        st.write(f"**Normal Cognitive Status (CDR 0.0):** `{prob_no_dementia:.1%}`")
        st.progress(float(prob_no_dementia))
        
        st.write(f"**Very Mild Impairment (CDR 0.5):** `{prob_very_mild:.1%}`")
        st.progress(float(prob_very_mild))
        
        st.write(f"**Clinical Dementia (CDR 1.0+):** `{prob_dementia:.1%}`")
        st.progress(float(prob_dementia))

    st.markdown("<div class='section-header'>3. Patient Parameter Audit Table</div>", unsafe_allow_html=True)
    st.caption("Summary of entered clinical parameters and corresponding model input values for clinical auditability.")

    audit_data = [
        {"Category": "Demographic", "Parameter": "Biological Sex (M/F)", "Entered Value": str(gender_input), "Model Value": str(gender_code), "Reference Standard": "Female (0) / Male (1)"},
        {"Category": "Demographic", "Parameter": "Age", "Entered Value": f"{age_input:.0f} years", "Model Value": f"{age_input:.1f}", "Reference Standard": "18 - 105 years"},
        {"Category": "Demographic", "Parameter": "Education Level (Educ)", "Entered Value": str(EDUC_OPTIONS[educ_level_input]), "Model Value": f"{educ_years:.0f} years", "Reference Standard": "Tier 1 - 5"},
        {"Category": "Demographic", "Parameter": "Socioeconomic Status (SES)", "Entered Value": f"Class {int(ses_input)}", "Model Value": f"{ses_input:.1f}", "Reference Standard": "1 (Highest) - 5 (Lowest)"},
        {"Category": "Cognitive", "Parameter": "MMSE Score", "Entered Value": f"{mmse_input:.0f} / 30", "Model Value": f"{mmse_input:.1f}", "Reference Standard": "0 - 30 (> 24 Normal)"},
        {"Category": "Volumetric", "Parameter": "Estimated Total Intracranial Volume (eTIV)", "Entered Value": f"{etiv_input:.0f} mm³", "Model Value": f"{etiv_input:.1f}", "Reference Standard": "800 - 2500 mm³"},
        {"Category": "Volumetric", "Parameter": "Normalized Whole Brain Volume (nWBV)", "Entered Value": f"{nwbv_input:.3f}", "Model Value": f"{nwbv_input:.3f}", "Reference Standard": "0.500 - 0.950"},
        {"Category": "Volumetric", "Parameter": "Atlas Scaling Factor (ASF)", "Entered Value": f"{asf_input:.3f}", "Model Value": f"{asf_input:.3f}", "Reference Standard": "0.700 - 1.800"}
    ]

    st.dataframe(pd.DataFrame(audit_data), hide_index=True)

    if profile["cdr"] is not None:
        st.markdown("<div class='section-header'>4. OASIS Ground-Truth Reference Verification</div>", unsafe_allow_html=True)
        expected_class = 0 if profile["cdr"] == 0.0 else (1 if profile["cdr"] == 0.5 else 2)
        match_status = predicted_class == expected_class
        
        with st.expander(f"Reference Benchmark Comparison: {profile['id']}", expanded=True):
            val_col1, val_col2, val_col3 = st.columns(3)
            with val_col1:
                st.metric("OASIS Subject ID", profile["id"])
                st.metric("Ground-Truth CDR Score", f"CDR {profile['cdr']}")
            with val_col2:
                st.metric("Ground-Truth Diagnosis", profile["ground_truth"])
                st.metric("Model Predicted Stage", stage_info["badge"])
            with val_col3:
                if match_status:
                    st.success("Diagnostic Concordance Confirmed: The model prediction aligns with the OASIS clinical reference assessment.")
                else:
                    st.info("Diagnostic Boundary Assessment: Prediction reflects borderline continuous parameter staging.")

# =============================================================================
# TAB 2: PATIENT RECORDS DATABASE
# =============================================================================
with tab_records:
    st.markdown("<div class='section-header'>Patient Records & Prediction History</div>", unsafe_allow_html=True)
    st.caption("Secure local clinical registry storing patient parameters and associated machine learning diagnostic staging.")
    
    try:
        records_df = get_all_records()
    except Exception as db_read_err:
        records_df = pd.DataFrame()
        st.error(f"Error retrieving database records: {db_read_err}")

    # Search & Filter Controls
    search_col1, search_col2 = st.columns([2, 1])
    with search_col1:
        search_query = st.text_input("Search by Patient ID, Name, or Diagnosis:", placeholder="e.g., OAS1_0001, DEMO, Normal...")
    with search_col2:
        date_filter = st.text_input("Filter by Date (YYYY-MM-DD):", placeholder="e.g., 2026-10")

    filtered_df = records_df.copy()
    if not filtered_df.empty:
        if search_query:
            q = search_query.strip().lower()
            mask = (
                filtered_df["patient_name"].astype(str).str.lower().str.contains(q, na=False) |
                filtered_df["patient_id"].astype(str).str.contains(q, na=False) |
                filtered_df["prediction_label"].astype(str).str.lower().str.contains(q, na=False)
            )
            filtered_df = filtered_df[mask]
            
        if date_filter:
            d = date_filter.strip()
            mask_date = filtered_df["created_at"].astype(str).str.contains(d, na=False)
            filtered_df = filtered_df[mask_date]

        # Display Data Table
        display_columns = [
            "patient_id", "patient_name", "created_at", "gender", "age",
            "educ_level", "ses", "mmse", "etiv", "nwbv", "asf",
            "prediction_label", "probability", "risk_level", "model_name"
        ]
        available_cols = [c for c in display_columns if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols].rename(columns={
                "patient_id": "Patient ID",
                "patient_name": "Identifier / Reference",
                "created_at": "Timestamp",
                "gender": "Sex",
                "age": "Age",
                "educ_level": "Educ",
                "ses": "SES",
                "mmse": "MMSE",
                "etiv": "eTIV",
                "nwbv": "nWBV",
                "asf": "ASF",
                "prediction_label": "Predicted Diagnosis",
                "probability": "Confidence",
                "risk_level": "Staging",
                "model_name": "Model"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Download CSV Section
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Patient Records (CSV)",
            data=csv_data,
            file_name="patient_records_export.csv",
            mime="text/csv"
        )
    else:
        st.info("No records found in database. Run a clinical prediction to record patient parameters.")

    st.markdown("---")
    st.markdown("<div class='category-header'>Manage Patient Records</div>", unsafe_allow_html=True)
    
    del_col1, del_col2 = st.columns([1, 2])
    with del_col1:
        patient_to_delete = st.number_input("Patient ID to Delete:", min_value=1, step=1, value=1)
        confirm_deletion = st.checkbox("Confirm deletion of patient record and cascading predictions")
        if st.button("Delete Patient Record"):
            if confirm_deletion:
                try:
                    deleted = delete_patient(patient_to_delete)
                    if deleted:
                        st.success(f"Patient ID #{patient_to_delete} and associated predictions successfully deleted.")
                        st.rerun()
                    else:
                        st.warning(f"Patient ID #{patient_to_delete} not found.")
                except Exception as del_err:
                    st.error(f"Error deleting record: {del_err}")
            else:
                st.warning("Please check the confirmation box before deleting.")
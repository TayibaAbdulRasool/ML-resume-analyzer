"""
ML Resume Analyzer — Streamlit App
-----------------------------------
Loads a pre-trained XGBoost model + StandardScaler (trained in the
accompanying notebook) and predicts whether a candidate would be
shortlisted, based on six resume/profile features.

Run with:
    streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config that MUST match the training notebook exactly
# --------------------------------------------------------------------------

MODEL_PATH = "resume_model.pkl"
SCALER_PATH = "scaler.pkl"

# Exact feature order used when the model/scaler were fit
FEATURE_ORDER = [
    "years_experience",
    "skills_match_score",
    "education_level",
    "project_count",
    "resume_length",
    "github_activity",
]

# LabelEncoder on education_level was fit alphabetically:
# Bachelors=0, High School=1, Masters=2, PhD=3
EDUCATION_ENCODING = {
    "High School": 1,
    "Bachelors": 0,
    "Masters": 2,
    "PhD": 3,
}

# Reasonable bounds (from the training data's describe()) — used only
# for input validation / slider ranges, not for any transformation.
FIELD_BOUNDS = {
    "years_experience": (0, 40),
    "skills_match_score": (0.0, 100.0),
    "project_count": (0, 60),
    "resume_length": (50, 2000),
    "github_activity": (0, 2000),
}

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="ML Resume Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #0a0a0a;
            --surface: #141414;
            --surface-2: #1e1a15;
            --border: rgba(255, 140, 0, 0.35);
            --orange: #ff8c00;
            --orange-bright: #ffa733;
            --text-muted: #b8a894;
        }

        .stApp, .main, body { background-color: var(--bg) !important; }
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 780px; }

        h1 {
            font-weight: 800;
            background: linear-gradient(90deg, #ffffff, var(--orange-bright));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: var(--text-muted); font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1.6rem; }

        h2, h3, h4, label, p, span, .stMarkdown { color: #f0f0f0; }

        /* Form container */
        div[data-testid="stForm"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px;
            padding: 1.6rem 1.6rem 0.8rem 1.6rem;
        }

        /* --- ALL input / select / textarea surfaces (catches every Streamlit version) --- */
        div[data-testid="stNumberInput"] div,
        div[data-testid="stTextInput"] div,
        div[data-testid="stSelectbox"] div,
        div[data-baseweb="input"],
        div[data-baseweb="base-input"],
        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"],
        ul[data-baseweb="menu"],
        li[role="option"],
        input, textarea, select {
            background-color: var(--surface-2) !important;
            color: #f5efe6 !important;
            border-color: var(--border) !important;
        }
        input, textarea { caret-color: var(--orange-bright) !important; }

        /* Number input +/- step buttons */
        div[data-testid="stNumberInput"] button {
            background-color: var(--surface-2) !important;
            border-color: var(--border) !important;
            color: var(--orange-bright) !important;
        }
        div[data-testid="stNumberInput"] button svg { fill: var(--orange-bright) !important; }

        /* Selectbox dropdown arrow / options */
        div[data-baseweb="select"] svg { fill: var(--orange-bright) !important; }
        li[role="option"]:hover, li[aria-selected="true"] {
            background-color: rgba(255, 140, 0, 0.18) !important;
        }

        div[data-testid="stWidgetLabel"] p { color: #e8dcc8 !important; font-weight: 500; }

        /* Slider */
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: var(--orange) !important;
            border-color: var(--orange) !important;
        }
        div[data-baseweb="slider"] > div > div { background: rgba(255, 140, 0, 0.25) !important; }
        div[data-baseweb="slider"] > div > div > div { background: var(--orange) !important; }
        div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] { color: var(--text-muted) !important; }
        div[data-testid="stThumbValue"] { color: var(--orange-bright) !important; }
        input[type="range"] { accent-color: var(--orange) !important; }

        /* Result cards */
        .result-card {
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            margin-top: 1.2rem;
            border: 1px solid var(--border);
            background-color: var(--surface);
        }
        .result-shortlisted {
            background: linear-gradient(135deg, #221604, #150e02);
            border: 1px solid rgba(255, 140, 0, 0.55);
        }
        .result-not-shortlisted {
            background: linear-gradient(135deg, #1a1a1a, #101010);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem; color: #ffffff; }
        .result-shortlisted .result-title { color: var(--orange-bright); }
        .result-sub { color: var(--text-muted); font-size: 0.95rem; }

        /* Submit button */
        div[data-testid="stFormSubmitButton"] button,
        .stButton > button {
            width: 100% !important;
            border-radius: 10px !important;
            padding: 0.7rem 0 !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            border: none !important;
            background: linear-gradient(90deg, #ff8c00, #ff6a00) !important;
            color: #0a0a0a !important;
            box-shadow: 0 4px 14px rgba(255, 140, 0, 0.25);
        }
        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button div,
        div[data-testid="stFormSubmitButton"] button span,
        .stButton > button p,
        .stButton > button div,
        .stButton > button span {
            color: #0a0a0a !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        .stButton > button:hover {
            background: linear-gradient(90deg, #ffa733, #ff8c00) !important;
            opacity: 1 !important;
        }
        div[data-testid="stFormSubmitButton"] button:disabled {
            background: #3a3a3a !important;
            color: #888 !important;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; color: var(--orange-bright) !important; }
        div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

        /* Progress bar */
        div[data-testid="stProgress"] div[role="progressbar"] > div { background-color: var(--orange) !important; }

        /* Expander */
        details {
            background-color: var(--surface) !important;
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        summary { color: var(--orange-bright) !important; }
        summary p { color: var(--orange-bright) !important; }

        hr { border-color: rgba(255, 140, 0, 0.2); }

        /* Help tooltip icon */
        svg[title], [data-testid="stTooltipIcon"] svg { fill: var(--orange-bright) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model / scaler loading (cached so files are read once per session)
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_artifacts():
    errors = []
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found: '{MODEL_PATH}'")
    if not os.path.exists(SCALER_PATH):
        errors.append(f"Scaler file not found: '{SCALER_PATH}'")
    if errors:
        return None, None, errors

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        errors.append(f"Failed to load model: {e}")
        model = None

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:  # noqa: BLE001
        errors.append(f"Failed to load scaler: {e}")
        scaler = None

    return model, scaler, errors


model, scaler, load_errors = load_artifacts()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("🧠 ML Resume Analyzer")
st.markdown(
    '<div class="subtitle">Predict whether a candidate is likely to be shortlisted, '
    "using a trained XGBoost classifier.</div>",
    unsafe_allow_html=True,
)

if load_errors:
    for err in load_errors:
        st.error(err)
    st.info(
        "Place **resume_model.pkl** and **scaler.pkl** in the same folder as this "
        "app (or update `MODEL_PATH` / `SCALER_PATH` above) and rerun the app."
    )
    st.stop()

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------

with st.form("resume_form"):
    st.subheader("Candidate Details")

    col1, col2 = st.columns(2)

    with col1:
        years_experience = st.number_input(
            "Years of Experience",
            min_value=FIELD_BOUNDS["years_experience"][0],
            max_value=FIELD_BOUNDS["years_experience"][1],
            value=5,
            step=1,
            help="Total professional experience, in years.",
        )
        education_level = st.selectbox(
            "Education Level",
            options=list(EDUCATION_ENCODING.keys()),
            index=1,
        )
        project_count = st.number_input(
            "Project Count",
            min_value=FIELD_BOUNDS["project_count"][0],
            max_value=FIELD_BOUNDS["project_count"][1],
            value=8,
            step=1,
            help="Number of relevant projects listed on the resume.",
        )

    with col2:
        skills_match_score = st.slider(
            "Skills Match Score",
            min_value=FIELD_BOUNDS["skills_match_score"][0],
            max_value=FIELD_BOUNDS["skills_match_score"][1],
            value=70.0,
            step=0.1,
            help="How well the candidate's skills match the job requirements (0–100).",
        )
        resume_length = st.number_input(
            "Resume Length (words)",
            min_value=FIELD_BOUNDS["resume_length"][0],
            max_value=FIELD_BOUNDS["resume_length"][1],
            value=550,
            step=10,
        )
        github_activity = st.number_input(
            "GitHub Activity (contributions)",
            min_value=FIELD_BOUNDS["github_activity"][0],
            max_value=FIELD_BOUNDS["github_activity"][1],
            value=300,
            step=5,
        )

    submitted = st.form_submit_button("🔍Analyze Resume")

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def build_feature_row(
    years_experience,
    skills_match_score,
    education_level,
    project_count,
    resume_length,
    github_activity,
):
    """Assemble a single-row DataFrame in the EXACT column order the
    scaler/model were trained on."""
    encoded_education = EDUCATION_ENCODING[education_level]
    row = {
        "years_experience": years_experience,
        "skills_match_score": skills_match_score,
        "education_level": encoded_education,
        "project_count": project_count,
        "resume_length": resume_length,
        "github_activity": github_activity,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def validate_inputs(values: dict) -> list:
    """Basic sanity checks beyond what the widgets already enforce."""
    problems = []
    for key, val in values.items():
        if val is None:
            problems.append(f"'{key}' is missing.")
            continue
        if isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val)):
            problems.append(f"'{key}' must be a finite number.")
    if values.get("skills_match_score") is not None and not (
        0 <= values["skills_match_score"] <= 100
    ):
        problems.append("'skills_match_score' must be between 0 and 100.")
    return problems


if submitted:
    raw_values = {
        "years_experience": years_experience,
        "skills_match_score": skills_match_score,
        "education_level": education_level,
        "project_count": project_count,
        "resume_length": resume_length,
        "github_activity": github_activity,
    }

    problems = validate_inputs(raw_values)

    if problems:
        for p in problems:
            st.error(p)
    else:
        try:
            features_df = build_feature_row(
                years_experience,
                skills_match_score,
                education_level,
                project_count,
                resume_length,
                github_activity,
            )

            scaled_features = scaler.transform(features_df)

            prediction = model.predict(scaled_features)[0]

            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(scaled_features)[0]

            is_shortlisted = int(prediction) == 1

            if is_shortlisted:
                confidence = proba[1] if proba is not None else None
                st.markdown(
                    f"""
                    <div class="result-card result-shortlisted">
                        <div class="result-title">✅ Shortlisted</div>
                        <div class="result-sub">
                            This candidate profile matches the pattern of previously
                            shortlisted candidates.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                confidence = proba[0] if proba is not None else None
                st.markdown(
                    f"""
                    <div class="result-card result-not-shortlisted">
                        <div class="result-title">❌ Not Shortlisted</div>
                        <div class="result-sub">
                            This candidate profile does not closely match previously
                            shortlisted candidates.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if proba is not None:
                st.write("")
                c1, c2 = st.columns(2)
                c1.metric("Probability — Shortlisted", f"{proba[1] * 100:.1f}%")
                c2.metric("Probability — Not Shortlisted", f"{proba[0] * 100:.1f}%")
                st.progress(float(proba[1]))

            with st.expander("See the exact feature vector sent to the model"):
                display_df = features_df.copy()
                display_df["education_level"] = display_df["education_level"].astype(str)
                display_df.loc[:, "education_level"] = (
                    f"{education_level} → {EDUCATION_ENCODING[education_level]}"
    )
                st.dataframe(display_df, use_container_width=True)
                st.caption(
                    "Values above are pre-scaling. The model receives these "
                    "features after being transformed by the saved StandardScaler."
                )

        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption(
    "Model: XGBoost classifier · Features are scaled with the training-time "
    "StandardScaler before inference. Predictions are estimates, not guarantees."
)
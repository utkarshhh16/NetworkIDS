# dashboad.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from io import BytesIO

# -----------------------
# Page config & styles
# -----------------------
st.set_page_config(page_title="CyberAI IDS — Interactive Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* overall */
body { background-color: #f6f8fb; }

/* header */
.header-box {
  background: linear-gradient(90deg,#0f172a 0%, #1e3a8a 100%);
  color: #ffffff;
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(16,24,40,0.08);
}

/* dataset info */
.dataset-info {
  background-color: #000000;
  padding: 12px;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(2,6,23,0.06);
}

/* metric card */
.metric-card {
  background: linear-gradient(180deg,#000000);
  padding: 12px;
  border-radius: 10px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(2,6,23,0.04);
  margin-bottom: 8px;
}

/* badge */
.small-muted { color:#6b7280; font-size:13px; }

/* probability bar label */
.prob-label { font-weight:600; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Helper functions
# -----------------------
def base_path_join(fname):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, fname)

@st.cache_resource
def load_artifacts_paths():
    """Load model and preprocessors using absolute paths (so working dir doesn't matter)."""
    try:
        model = joblib.load(base_path_join("rf_intrusion_model.pkl"))
        scaler = joblib.load(base_path_join("scaler.pkl"))
        encoder = joblib.load(base_path_join("label_encoder.pkl"))
        return model, scaler, encoder
    except Exception as e:
        raise

def detect_label_column(df):
    """Try to detect the true label column in uploaded dataset"""
    candidates = ['label', 'Label', 'target', 'Target', 'class', 'Class', 'attack', 'Attack', 'y', 'Y']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: if last column looks categorical with few unique values, return it
    last = df.columns[-1]
    if df[last].nunique() <= 10:
        return last
    return None

def prepare_features_from_df(df, selected_features):
    """Return a numeric-only dataframe ordered as model expects (selected_features)."""
    X = df[selected_features].copy()
    # try to convert to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.isnull().values.any():
        # inform user but continue (they may have non-numeric columns)
        st.warning("Some feature columns had non-numeric values and were coerced to NaN. Please ensure features are numeric and in the correct order.")
    return X

# -----------------------
# Load artifacts
# -----------------------
st.markdown('<div class="header-box"><h2>CyberAI — Intrusion Detection Dashboard</h2><div class="small-muted">Interactive simulation • model-driven outputs • judge-friendly visuals</div></div>', unsafe_allow_html=True)
st.write("")  # spacing

try:
    model, scaler, encoder = load_artifacts_paths()
    st.success("Model & preprocessors loaded from repo folder.")
except Exception as e:
    st.error("Failed to load model/scaler/encoder. Make sure rf_intrusion_model.pkl, scaler.pkl and label_encoder.pkl are in the same folder as this script.")
    st.exception(e)
    st.stop()

# -----------------------
# Data loading / upload
# -----------------------
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("### Dataset")
    uploaded = st.file_uploader("Upload CSV (optional). If not provided the app will attempt to use 'samples.csv' or first CSV in ./Data/.", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            data_source = f"Uploaded: {getattr(uploaded,'name', 'uploaded.csv')}"
        except Exception as e:
            st.error("Failed to read uploaded CSV. Please ensure it is a valid comma-separated file.")
            st.stop()
    else:
        # fall back
        sample_path = base_path_join("samples.csv")
        data = None
        if os.path.exists(sample_path):
            try:
                data = pd.read_csv(sample_path)
                data_source = "Local: samples.csv"
            except:
                data = None
        else:
            # check Data folder
            data_dir = base_path_join("Data")
            if os.path.isdir(data_dir):
                csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
                if csvs:
                    try:
                        data = pd.read_csv(os.path.join(data_dir, csvs[0]))
                        data_source = f"Local: Data/{csvs[0]}"
                    except:
                        data = None
        if data is None:
            st.warning("No dataset found. Upload a CSV or place a 'samples.csv' file in the same folder or put a CSV in ./Data/. The simulate button will be disabled until a dataset is available.")
            data = None

with col2:
    st.markdown("### Active Source")
    if data is not None:
        st.markdown(f'<div class="dataset-info"><b>{data_source}</b><br><span class="small-muted">Rows: {data.shape[0]} • Columns: {data.shape[1]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="dataset-info"><b>No dataset loaded</b><br><span class="small-muted">Upload or add samples.csv to enable simulation.</span></div>', unsafe_allow_html=True)

# If no data, stop after showing source info
if data is None:
    st.stop()

# -----------------------
# Detect label column and features
# -----------------------
label_col = detect_label_column(data)
st.markdown("#### Columns preview")
st.dataframe(pd.DataFrame({"Columns": list(data.columns)}), height=160)

# Let user pick features (ordered)
st.markdown("#### Feature selection (choose columns used as model input and ensure order matches training)")
# By default, suggest all numeric columns except detected label
suggested = [c for c in data.columns if c != label_col]
selected = st.multiselect("Select feature columns (order matters). Use drag to reorder after selection.", options=suggested, default=suggested)

if not selected:
    st.warning("Select at least one feature column to proceed.")
    st.stop()

# -----------------------
# Prepare X and y (if available)
# -----------------------
X_raw = prepare_features_from_df(data, selected)

y_true = None
if label_col:
    y_true = data[label_col].copy()
    # If labels are encoded integers but your encoder expects e.g. names, try to align:
    st.info(f"Detected label column: `{label_col}` — using it for evaluation (if values match encoding).")

# -----------------------
# Simulate button & prediction
# -----------------------
st.markdown("---")
st.markdown("### Simulation")
colA, colB = st.columns([3,2])

with colA:
    sim_btn = st.button("🎯 Simulate random sample and predict")

with colB:
    st.markdown("### Quick actions")
    st.write("")
    if st.button("🔁 Refresh preview"):
        st.experimental_rerun()

if not sim_btn:
    st.info("Click **Simulate random sample and predict** to pick a random record, run inference, and view model outputs.")
    st.stop()

# Choose random row index
idx = np.random.randint(0, X_raw.shape[0])
sample_raw = X_raw.iloc[[idx]].reset_index(drop=True)
sample_original = data.iloc[[idx]].reset_index(drop=True)
st.markdown("#### Selected Sample (original values)")
st.dataframe(sample_original, height=160)

# Scale features
try:
    X_scaled = scaler.transform(sample_raw)
except Exception as e:
    st.error("Scaler failed on selected sample. Ensure your selected features match the model training features and are numeric.")
    st.exception(e)
    st.stop()

# Predict
pred_proba = model.predict_proba(X_scaled)[0]
pred_idx = np.argmax(pred_proba)
pred_label = encoder.inverse_transform([pred_idx])[0] if hasattr(encoder, "inverse_transform") else model.classes_[pred_idx]
# Note: encoder.inverse_transform expects encoded labels; above we used index mapping to be safe.

# Build ordered class/prob table
classes = list(encoder.classes_) if hasattr(encoder, "classes_") else list(model.classes_)
# Map probabilities to classes correctly: model.classes_ gives correct order for predict_proba
if hasattr(model, "classes_"):
    classes = list(model.classes_)
# If encoder maps numeric to text labels, try to inverse map model.classes_
try:
    # if model.classes_ are encoded integers, convert to readable labels via encoder
    if hasattr(encoder, "inverse_transform") and np.issubdtype(model.classes_.dtype, np.integer):
        readable_classes = encoder.inverse_transform(model.classes_)
    else:
        readable_classes = np.array(classes)
except Exception:
    readable_classes = np.array(classes)

probs_df = pd.DataFrame({"class": readable_classes, "prob": pred_proba})
probs_df = probs_df.sort_values("prob", ascending=False).reset_index(drop=True)

# -----------------------
# Output Panel — Prediction + explanation
# -----------------------
st.markdown("#### Prediction Summary")

col1, col2, col3 = st.columns([3,2,2])
with col1:
    # big label & confidence
    top_label = probs_df.loc[0, "class"]
    top_prob = probs_df.loc[0, "prob"] * 100
    st.markdown(f"<div style='padding:12px;border-radius:10px;background:#000;box-shadow:0 4px 20px rgba(2,6,23,0.06)'><h3 style='margin:0'>{'✅ Normal' if any(w in str(top_label).lower() for w in ['normal']) else '🚨 Attack Detected'}</h3><div style='color:#6b7280'>Predicted class: <b>{top_label}</b></div><div style='margin-top:8px;font-size:18px'><b>Confidence: {top_prob:.2f}%</b></div></div>", unsafe_allow_html=True)
    # suggested mitigation
    response_map = {
        "dos": "Block or rate-limit high-traffic IPs; apply rate-limiting.",
        "ddos": "Activate DDoS protection at edge; filter malicious traffic.",
        "probe": "Investigate source IP and scan logs for reconnaissance activity.",
        "r2l": "Monitor failed logins, reset credentials, restrict remote access.",
        "u2r": "Review privilege escalation logs, patch vulnerable services.",
        "normal": "No immediate action; continue standard monitoring.",
        "malware": "Quarantine affected hosts and run AV/forensic analysis."
    }
    found = False
    for k, v in response_map.items():
        if k in str(top_label).lower():
            st.info(f"Suggested action: {v}")
            found = True
            break
    if not found:
        st.info("Suggested action: Review logs for anomalies and follow incident response procedure.")

with col2:
    # show top-2 comparison in a compact card
    st.markdown("<div class='metric-card'><h4 style='margin:0'>Top 2 Classes</h4></div>", unsafe_allow_html=True)
    st.write(f"1. **{probs_df.loc[0,'class']}** — {probs_df.loc[0,'prob']*100:.2f}%")
    if len(probs_df) > 1:
        st.write(f"2. {probs_df.loc[1,'class']} — {probs_df.loc[1,'prob']*100:.2f}%")

with col3:
    # progress gauge (visual)
    st.markdown("<div class='metric-card'><h4 style='margin:0'>Confidence Gauge</h4></div>", unsafe_allow_html=True)
    st.progress(int(top_prob))

# -----------------------
# Class probabilities chart (altair)
# -----------------------
st.markdown("#### Class probabilities")
chart = alt.Chart(probs_df).mark_bar().encode(
    x=alt.X('prob:Q', axis=alt.Axis(format='.0%', title='Probability')),
    y=alt.Y('class:N', sort='-x', title='Class'),
    tooltip=[alt.Tooltip('class:N'), alt.Tooltip('prob:Q', format='.2%')]
).properties(height=200, width=800)
st.altair_chart(chart, use_container_width=True)

# -----------------------
# If true labels available — compute metrics & confusion matrix
# -----------------------
metrics_displayed = False
if y_true is not None:
    # try to transform y_true to same format as encoder expects
    try:
        # if encoder expects strings and y_true is numeric codes, try inverse_transform
        # but more robust: predict on full X and compare after mapping to readable labels
        X_all = prepare_features_from_df(data, selected)
        X_all_scaled = scaler.transform(X_all)
        y_pred_all_idxs = model.predict(X_all_scaled)
        # model.classes_ may be encoded values; try to map to readable labels
        try:
            y_pred_readable = encoder.inverse_transform(y_pred_all_idxs)
        except Exception:
            # fallback: if y_true is already readable, compare directly
            y_pred_readable = y_pred_all_idxs
        # use y_true as-is if matches values; else try to map numeric codes
        y_true_values = y_true.values
        # compute metrics only if shapes align
        if len(y_true_values) == len(y_pred_readable):
            acc = accuracy_score(y_true_values, y_pred_readable)
            prec = precision_score(y_true_values, y_pred_readable, average='macro', zero_division=0)
            rec = recall_score(y_true_values, y_pred_readable, average='macro', zero_division=0)
            f1 = f1_score(y_true_values, y_pred_readable, average='macro', zero_division=0)
            st.markdown("### Model Evaluation on provided dataset")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc*100:.2f}%")
            c2.metric("Precision (macro)", f"{prec*100:.2f}%")
            c3.metric("Recall (macro)", f"{rec*100:.2f}%")
            c4.metric("F1 (macro)", f"{f1*100:.2f}%")
            # confusion matrix (compact)
            cm = confusion_matrix(y_true_values, y_pred_readable, labels=np.unique(np.concatenate([y_true_values, y_pred_readable])))
            cm_df = pd.DataFrame(cm, index=np.unique(np.concatenate([y_true_values, y_pred_readable])),
                                 columns=np.unique(np.concatenate([y_true_values, y_pred_readable])))
            st.markdown("#### Confusion matrix (rows=true, cols=pred)")
            st.dataframe(cm_df.style.background_gradient(cmap='Blues'), height=260)
            metrics_displayed = True
    except Exception as ex:
        st.warning("Could not compute full evaluation on dataset automatically. Ensure label values match model's label encoding.")
        st.exception(ex)

# If no dataset labels and model has oob_score_, show it
if (not metrics_displayed) and hasattr(model, "oob_score_"):
    st.markdown("### Model metric (OOB estimate)")
    st.metric("Out-of-bag accuracy", f"{model.oob_score_*100:.2f}%")

# -----------------------
# Top features (if available) — compact textual list, not a heavy static plot
# -----------------------
if hasattr(model, "feature_importances_"):
    try:
        imps = model.feature_importances_
        # align with selected features
        feat_imp_df = pd.DataFrame({"feature": selected, "importance": imps[:len(selected)]})
        feat_imp_df = feat_imp_df.sort_values("importance", ascending=False).head(8)
        st.markdown("#### Top feature signals (from model)")
        st.table(feat_imp_df.reset_index(drop=True))
    except Exception:
        # ignore if something odd
        pass

st.markdown("---")
st.markdown("Help: If predictions look wrong, ensure `selected features` exactly match the training features in both **order** and **type**. Use the uploader to test other sample files.")

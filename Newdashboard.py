# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from io import BytesIO
from datetime import datetime
import json

# -----------------------
# EXPECTED FEATURES (from model training)
# -----------------------
EXPECTED_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Average Packet Size", "Subflow Fwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Max", "Active Min",
    "Idle Mean", "Idle Max", "Idle Min"
]

# -----------------------
# Page config & styles
# -----------------------
st.set_page_config(page_title="CyberAI IDS – Interactive Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
body { background-color: #f6f8fb; }

.header-box {
  background: linear-gradient(90deg,#0f172a 0%, #1e3a8a 100%);
  color: #ffffff;
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(16,24,40,0.08);
}

.dataset-info {
  background-color: #000000;
  padding: 12px;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(2,6,23,0.06);
  color: #ffffff;
}

.metric-card {
  background: linear-gradient(180deg,#000000);
  padding: 12px;
  border-radius: 10px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(2,6,23,0.04);
  margin-bottom: 8px;
  color: #ffffff;
}

.small-muted { color:#6b7280; font-size:13px; }
.prob-label { font-weight:600; margin-bottom:6px; }
.success-box { background-color: #dcfce7; padding: 10px; border-radius: 8px; border-left: 4px solid #16a34a; }
.error-box { background-color: #fee2e2; padding: 10px; border-radius: 8px; border-left: 4px solid #dc2626; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Session state initialization
# -----------------------
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'data_loaded_time' not in st.session_state:
    st.session_state.data_loaded_time = None

# -----------------------
# Helper functions
# -----------------------
def base_path_join(fname):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, fname)

@st.cache_resource
def load_artifacts_paths():
    """Load model and preprocessors using absolute paths."""
    try:
        model = joblib.load(base_path_join("models/rf_intrusion_model.pkl"))
        scaler = joblib.load(base_path_join("models/scaler.pkl"))
        encoder = joblib.load(base_path_join("models/label_encoder.pkl"))
        return model, scaler, encoder
    except Exception as e:
        raise

def detect_label_column(df):
    """Try to detect the true label column in uploaded dataset"""
    candidates = ['label', 'Label', 'target', 'Target', 'class', 'Class', 'attack', 'Attack', 'y', 'Y']
    for c in candidates:
        if c in df.columns:
            return c
    last = df.columns[-1]
    if df[last].nunique() <= 10:
        return last
    return None

def validate_features(df):
    """Check if dataframe contains all expected features and return status"""
    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_FEATURES and c != detect_label_column(df)]
    return missing, extra

def prepare_features_from_df(df, required_features):
    """Return a numeric-only dataframe ordered as model expects."""
    try:
        X = df[required_features].copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        if X.isnull().values.any():
            nan_cols = X.columns[X.isnull().any()].tolist()
            st.warning(f"⚠️ Some features have non-numeric values and were coerced to NaN: {', '.join(nan_cols)}")
        return X
    except KeyError as e:
        st.error(f"❌ Missing required features: {e}")
        return None

def add_to_history(sample_idx, pred_label, confidence, true_label=None):
    """Add prediction to history"""
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sample_idx': sample_idx,
        'predicted': pred_label,
        'confidence': f"{confidence:.2f}%",
        'true_label': true_label if true_label is not None else 'N/A'
    }
    st.session_state.prediction_history.append(entry)
    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history.pop(0)

# -----------------------
# Load artifacts
# -----------------------
st.markdown('<div class="header-box"><h2>🛡️ CyberAI – Intrusion Detection Dashboard</h2><div class="small-muted">Interactive simulation • Model-driven outputs • Advanced analytics</div></div>', unsafe_allow_html=True)
st.write("")

try:
    model, scaler, encoder = load_artifacts_paths()
    st.success("✅ Model & preprocessors loaded successfully from models folder.")
except Exception as e:
    st.error("❌ Failed to load model/scaler/encoder. Make sure rf_intrusion_model.pkl, scaler.pkl and label_encoder.pkl are in the models/ folder.")
    st.exception(e)
    st.stop()

# -----------------------
# Data loading / upload
# -----------------------
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("### 📊 Dataset")
    uploaded = st.file_uploader("Upload CSV (optional). If not provided, the app will use 'samples.csv' or first CSV in ./Data/", type=["csv"])
    
    data = None
    data_source = None
    
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            data_source = f"Uploaded: {getattr(uploaded,'name', 'uploaded.csv')}"
            st.session_state.data_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error("❌ Failed to read uploaded CSV. Please ensure it is a valid comma-separated file.")
            st.stop()
    else:
        sample_path = base_path_join("samples.csv")
        if os.path.exists(sample_path):
            try:
                data = pd.read_csv(sample_path)
                data_source = "Local: samples.csv"
                if st.session_state.data_loaded_time is None:
                    st.session_state.data_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except:
                data = None
        else:
            data_dir = base_path_join("Data")
            if os.path.isdir(data_dir):
                csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
                if csvs:
                    try:
                        data = pd.read_csv(os.path.join(data_dir, csvs[0]))
                        data_source = f"Local: Data/{csvs[0]}"
                        if st.session_state.data_loaded_time is None:
                            st.session_state.data_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        data = None

with col2:
    st.markdown("### 📁 Active Source")
    if data is not None:
        st.markdown(f'<div class="dataset-info"><b>{data_source}</b><br><span class="small-muted">Rows: {data.shape[0]:,} • Columns: {data.shape[1]}</span><br><span class="small-muted">Loaded: {st.session_state.data_loaded_time}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="dataset-info"><b>No dataset loaded</b><br><span class="small-muted">Upload or add samples.csv to enable simulation.</span></div>', unsafe_allow_html=True)

if data is None:
    st.warning("⚠️ No dataset found. Upload a CSV or place a 'samples.csv' file in the same folder or put a CSV in ./Data/")
    st.stop()

# -----------------------
# Validate features
# -----------------------
label_col = detect_label_column(data)
missing_features, extra_features = validate_features(data)

if missing_features:
    st.error(f"❌ Dataset is missing required features: {', '.join(missing_features[:5])}{'...' if len(missing_features) > 5 else ''}")
    st.info("💡 The model expects exactly these features in this order. Please ensure your dataset contains all required columns.")
    with st.expander("📋 View all expected features"):
        st.write(EXPECTED_FEATURES)
    st.stop()

if extra_features and len(extra_features) > 0:
    st.info(f"ℹ️ Dataset contains {len(extra_features)} extra columns (will be ignored): {', '.join(extra_features[:3])}{'...' if len(extra_features) > 3 else ''}")

# Show feature status
with st.expander("🔍 Feature Validation Status"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Expected Features (All Present)**")
        st.dataframe(pd.DataFrame({"Feature": EXPECTED_FEATURES}), height=300)
    with col2:
        st.markdown("**📊 Dataset Columns**")
        col_status = []
        for col in data.columns:
            if col in EXPECTED_FEATURES:
                col_status.append({"Column": col, "Status": "✅ Required"})
            elif col == label_col:
                col_status.append({"Column": col, "Status": "🏷️ Label"})
            else:
                col_status.append({"Column": col, "Status": "ℹ️ Extra"})
        st.dataframe(pd.DataFrame(col_status), height=300)

# -----------------------
# Prepare data
# -----------------------
X_raw = prepare_features_from_df(data, EXPECTED_FEATURES)
if X_raw is None:
    st.stop()

y_true = None
if label_col:
    y_true = data[label_col].copy()
    st.success(f"✅ Label column detected: `{label_col}` – Will use for evaluation metrics")

# -----------------------
# Quick Actions Bar
# -----------------------
st.markdown("---")
st.markdown("### ⚡ Quick Actions")
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    if st.button("🔄 Reload Dataset", help="Reload the current dataset from file"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.data_loaded_time = None
        st.rerun()

with col_b:
    if st.button("📜 View History", help="Show prediction history"):
        st.session_state.show_history = not st.session_state.get('show_history', False)

with col_c:
    if st.button("🧹 Clear History", help="Clear all prediction history"):
        st.session_state.prediction_history = []
        st.success("History cleared!")
        st.rerun()

with col_d:
    batch_mode = st.checkbox("📦 Batch Mode", help="Enable batch prediction on multiple samples")

# Show prediction history if toggled
if st.session_state.get('show_history', False) and len(st.session_state.prediction_history) > 0:
    with st.expander("📜 Prediction History (Last 50)", expanded=True):
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, height=200, use_container_width=True)
        
        # Download history
        csv = hist_df.to_csv(index=False)
        st.download_button(
            label="💾 Download History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# -----------------------
# Model Performance Metrics (Optional - Cached)
# -----------------------
@st.cache_data
def calculate_model_metrics(_model, _scaler, _encoder, X_data, y_data):
    """Calculate model metrics once and cache results"""
    try:
        X_all_scaled = _scaler.transform(X_data)
        y_pred_all = _model.predict(X_all_scaled)
        
        # Get readable labels
        try:
            if hasattr(_encoder, 'inverse_transform'):
                y_pred_readable = _encoder.inverse_transform(y_pred_all)
            else:
                y_pred_readable = y_pred_all
        except:
            y_pred_readable = y_pred_all
        
        y_true_values = y_data.values
        
        # Calculate metrics
        if len(y_true_values) == len(y_pred_readable):
            metrics = {
                'accuracy': accuracy_score(y_true_values, y_pred_readable),
                'precision_macro': precision_score(y_true_values, y_pred_readable, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true_values, y_pred_readable, average='weighted', zero_division=0),
                'recall_macro': recall_score(y_true_values, y_pred_readable, average='macro', zero_division=0),
                'recall_weighted': recall_score(y_true_values, y_pred_readable, average='weighted', zero_division=0),
                'f1_macro': f1_score(y_true_values, y_pred_readable, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_true_values, y_pred_readable, average='weighted', zero_division=0),
                'y_true': y_true_values,
                'y_pred': y_pred_readable
            }
            return metrics
        return None
    except Exception as ex:
        return None

if y_true is not None:
    st.markdown("---")
    st.markdown("### 📈 Model Performance Metrics")
    
    # Option to show metrics
    show_metrics = st.checkbox("📊 Calculate & Show Full Dataset Metrics (may take time on large datasets)", value=False)
    
    if show_metrics:
        with st.spinner("Calculating metrics on entire dataset..."):
            metrics = calculate_model_metrics(model, scaler, encoder, X_raw, y_true)
        
        if metrics:
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%")
            col2.metric("🎯 F1 Score (Macro)", f"{metrics['f1_macro']*100:.2f}%")
            col3.metric("🎯 Precision (Macro)", f"{metrics['precision_macro']*100:.2f}%")
            col4.metric("🎯 Recall (Macro)", f"{metrics['recall_macro']*100:.2f}%")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("⚖️ F1 Score (Weighted)", f"{metrics['f1_weighted']*100:.2f}%")
            col6.metric("⚖️ Precision (Weighted)", f"{metrics['precision_weighted']*100:.2f}%")
            col7.metric("⚖️ Recall (Weighted)", f"{metrics['recall_weighted']*100:.2f}%")
            
            if hasattr(model, 'oob_score_'):
                col8.metric("🌲 OOB Score", f"{model.oob_score_*100:.2f}%")
            else:
                col8.metric("📊 Samples", f"{len(metrics['y_true']):,}")
            
            # Confusion Matrix
            with st.expander("🔍 Confusion Matrix & Classification Report"):
                col_cm1, col_cm2 = st.columns([1, 1])
                
                with col_cm1:
                    st.markdown("**Confusion Matrix**")
                    unique_labels = np.unique(np.concatenate([metrics['y_true'], metrics['y_pred']]))
                    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'], labels=unique_labels)
                    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
                    st.dataframe(cm_df.style.background_gradient(cmap='Blues'), height=300)
                
                with col_cm2:
                    st.markdown("**Classification Report**")
                    report = classification_report(metrics['y_true'], metrics['y_pred'], output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['precision', 'recall', 'f1-score']), height=300)
        else:
            st.error("⚠️ Could not compute evaluation metrics. Ensure label values match model's expected format.")
    else:
        st.info("💡 Check the box above to calculate model performance on the full dataset (only calculated once and cached)")
else:
    # Show OOB if available
    if hasattr(model, 'oob_score_'):
        st.markdown("---")
        st.info(f"🌲 Model Out-of-Bag Score: **{model.oob_score_*100:.2f}%**")

# -----------------------
# Batch Prediction Mode
# -----------------------
if batch_mode:
    st.markdown("---")
    st.markdown("### 📦 Batch Prediction")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        num_samples = st.number_input("Number of samples to predict", min_value=1, max_value=min(100, len(X_raw)), value=min(10, len(X_raw)))
    with col2:
        sample_method = st.radio("Sampling method", ["Random", "First N", "Last N"], horizontal=True)
    
    if st.button("🚀 Run Batch Prediction"):
        with st.spinner("Processing batch predictions..."):
            # Select samples
            if sample_method == "Random":
                indices = np.random.choice(len(X_raw), size=num_samples, replace=False)
            elif sample_method == "First N":
                indices = np.arange(num_samples)
            else:  # Last N
                indices = np.arange(len(X_raw) - num_samples, len(X_raw))
            
            # Predict
            X_batch = X_raw.iloc[indices]
            X_batch_scaled = scaler.transform(X_batch)
            y_pred_batch = model.predict(X_batch_scaled)
            y_proba_batch = model.predict_proba(X_batch_scaled)
            
            # Get readable predictions
            try:
                if hasattr(encoder, 'inverse_transform'):
                    y_pred_readable = encoder.inverse_transform(y_pred_batch)
                else:
                    y_pred_readable = y_pred_batch
            except:
                y_pred_readable = y_pred_batch
            
            # Create results dataframe
            results = pd.DataFrame({
                'Sample_Index': indices,
                'Predicted_Class': y_pred_readable,
                'Confidence': [np.max(proba) * 100 for proba in y_proba_batch]
            })
            
            if y_true is not None:
                results['True_Label'] = y_true.iloc[indices].values
                results['Correct'] = results['Predicted_Class'] == results['True_Label']
            
            st.success(f"✅ Batch prediction completed for {num_samples} samples!")
            st.dataframe(results, height=300, use_container_width=True)
            
            # Add to history
            for idx, pred, conf in zip(indices, y_pred_readable, [np.max(proba) * 100 for proba in y_proba_batch]):
                true_val = y_true.iloc[idx] if y_true is not None else None
                add_to_history(int(idx), pred, conf, true_val)
            
            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="💾 Download Batch Results",
                data=csv,
                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Summary stats
            if y_true is not None:
                batch_acc = (results['Correct'].sum() / len(results)) * 100
                st.metric("Batch Accuracy", f"{batch_acc:.2f}%")

# -----------------------
# Single Sample Prediction
# -----------------------
if not batch_mode:
    st.markdown("---")
    st.markdown("### 🎯 Single Sample Prediction")
    
    colA, colB = st.columns([3, 2])
    with colA:
        sim_btn = st.button("🎲 Simulate Random Sample & Predict", use_container_width=True)
    with colB:
        manual_idx = st.number_input("Or select sample index manually", min_value=0, max_value=len(X_raw)-1, value=0)
        manual_btn = st.button("🔍 Predict Selected Sample", use_container_width=True)
    
    if not (sim_btn or manual_btn):
        st.info("👆 Click **Simulate Random Sample** to pick a random record, or enter a specific index to predict that sample.")
        st.stop()
    
    # Choose sample
    if sim_btn:
        idx = np.random.randint(0, X_raw.shape[0])
    else:
        idx = manual_idx
    
    sample_raw = X_raw.iloc[[idx]].reset_index(drop=True)
    sample_original = data.iloc[[idx]].reset_index(drop=True)
    
    st.markdown(f"#### 📋 Selected Sample (Index: {idx})")
    st.dataframe(sample_original, height=160, use_container_width=True)
    
    # Scale and predict
    try:
        X_scaled = scaler.transform(sample_raw)
        pred_proba = model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(pred_proba)
        
        # Get readable label
        if hasattr(model, 'classes_'):
            classes = model.classes_
            if hasattr(encoder, 'inverse_transform'):
                try:
                    readable_classes = encoder.inverse_transform(classes)
                except:
                    readable_classes = classes
            else:
                readable_classes = classes
        else:
            readable_classes = encoder.classes_ if hasattr(encoder, 'classes_') else ['Unknown']
        
        pred_label = readable_classes[pred_idx]
        
        # Build probability dataframe
        probs_df = pd.DataFrame({"class": readable_classes, "prob": pred_proba})
        probs_df = probs_df.sort_values("prob", ascending=False).reset_index(drop=True)
        
        # Add to history
        true_label_val = y_true.iloc[idx] if y_true is not None else None
        add_to_history(idx, pred_label, probs_df.loc[0, 'prob'] * 100, true_label_val)
        
    except Exception as e:
        st.error("❌ Prediction failed. Ensure data format matches model requirements.")
        st.exception(e)
        st.stop()
    
    # -----------------------
    # Prediction Results
    # -----------------------
    st.markdown("#### 🎯 Prediction Summary")
    
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        top_label = probs_df.loc[0, "class"]
        top_prob = probs_df.loc[0, "prob"] * 100
        
        is_normal = any(w in str(top_label).lower() for w in ['normal', 'benign'])
        status_emoji = "✅" if is_normal else "🚨"
        status_text = "Normal Traffic" if is_normal else "Attack Detected"
        bg_color = "#dcfce7" if is_normal else "#fee2e2"
        border_color = "#16a34a" if is_normal else "#dc2626"
        
        st.markdown(f"""
        <div style='padding:16px;border-radius:10px;background:{bg_color};border-left:4px solid {border_color};box-shadow:0 4px 20px rgba(0,0,0,0.1)'>
            <h2 style='margin:0;color:#1f2937'>{status_emoji} {status_text}</h2>
            <div style='color:#4b5563;margin-top:8px'>Predicted class: <b>{top_label}</b></div>
            <div style='margin-top:12px;font-size:24px;font-weight:bold;color:#1f2937'>Confidence: {top_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggested mitigation
        response_map = {
            "dos": "🛑 Block or rate-limit source IP; apply DDoS mitigation rules.",
            "ddos": "🛡️ Activate DDoS protection at network edge; contact ISP if needed.",
            "probe": "🔍 Investigate source IP and review firewall logs for reconnaissance activity.",
            "r2l": "🔐 Monitor failed logins; reset credentials; restrict remote access.",
            "u2r": "⚠️ Review privilege escalation logs; patch vulnerable services immediately.",
            "normal": "✅ No immediate action required. Continue standard monitoring.",
            "malware": "🦠 Quarantine affected hosts; run antivirus and forensic analysis."
        }
        
        action_found = False
        for k, v in response_map.items():
            if k in str(top_label).lower():
                st.info(f"**Recommended Action:** {v}")
                action_found = True
                break
        
        if not action_found:
            st.info("**Recommended Action:** Review security logs and follow incident response procedures.")
        
        # Show true label if available
        if y_true is not None:
            true_val = y_true.iloc[idx]
            is_correct = str(true_val).lower() == str(pred_label).lower()
            if is_correct:
                st.success(f"✅ **Correct Prediction!** True label: {true_val}")
            else:
                st.error(f"❌ **Incorrect Prediction.** True label: {true_val}")
    
    with col2:
        st.markdown("<div class='metric-card'><h4 style='margin:0'>Top 3 Classes</h4></div>", unsafe_allow_html=True)
        for i in range(min(3, len(probs_df))):
            emoji = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
            st.write(f"{emoji} **{probs_df.loc[i,'class']}** – {probs_df.loc[i,'prob']*100:.2f}%")
    
    with col3:
        st.markdown("<div class='metric-card'><h4 style='margin:0'>Confidence Gauge</h4></div>", unsafe_allow_html=True)
        st.progress(min(int(top_prob), 100))
        st.caption(f"Confidence Level: {'High' if top_prob > 80 else ('Medium' if top_prob > 50 else 'Low')}")
    
    # -----------------------
    # Probability Distribution Chart
    # -----------------------
    st.markdown("#### 📊 Class Probability Distribution")
    chart = alt.Chart(probs_df).mark_bar(color='steelblue').encode(
        x=alt.X('prob:Q', axis=alt.Axis(format='.0%', title='Probability'), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('class:N', sort='-x', title='Class'),
        tooltip=[
            alt.Tooltip('class:N', title='Class'),
            alt.Tooltip('prob:Q', format='.2%', title='Probability')
        ]
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)

# -----------------------
# Feature Importance
# -----------------------
if hasattr(model, "feature_importances_"):
    st.markdown("---")
    st.markdown("### 🔬 Feature Importance Analysis")
    
    try:
        imps = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "feature": EXPECTED_FEATURES[:len(imps)],
            "importance": imps[:len(EXPECTED_FEATURES)]
        })
        feat_imp_df = feat_imp_df.sort_values("importance", ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Top 15 Most Important Features**")
            chart = alt.Chart(feat_imp_df.head(15)).mark_bar().encode(
                x=alt.X('importance:Q', title='Importance Score'),
                y=alt.Y('feature:N', sort='-x', title='Feature'),
                color=alt.Color('importance:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                tooltip=['feature', alt.Tooltip('importance:Q', format='.4f')]
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Features (Table)**")
            st.dataframe(
                feat_imp_df.head(10).reset_index(drop=True),
                height=400,
                use_container_width=True
            )
    except Exception as e:
        st.warning("Could not display feature importance.")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#6b7280;padding:20px'>
    <b>CyberAI Intrusion Detection System</b> | Powered by Random Forest ML Model<br>
    💡 <i>Tip: All features must match training data exactly. Use batch mode for multiple predictions.</i>
</div>
""", unsafe_allow_html=True)
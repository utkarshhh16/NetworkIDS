import streamlit as st
import pandas as pd
import requests
import time

# Sidebar
st.sidebar.title("Control Panel")
rate = st.sidebar.slider("Replay rate (flows/sec)", min_value=1, max_value=500, value=50)
scenario = st.sidebar.selectbox("Demo Scenario", ["Normal", "PortScan", "DDoS", "BruteForce"])
start_button = st.sidebar.button("Start Replay")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model info**")
st.sidebar.text("XGBoost v1.0 + AE hybrid")
st.sidebar.markdown("---")

# Main layout: two columns
left, right = st.columns([2, 1])

with left:
    st.header("Live Flow Stream")
    flow_feed = st.empty()
    alert_table = st.empty()

with right:
    st.header("Alert Details")
    alert_id = st.empty()
    alert_info = st.empty()

st.markdown("---")
st.header("Explainability")
shap_box = st.empty()

# Small helper to render a flow dataframe
def render_flows(flows):
    df = pd.DataFrame(flows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# Polling buffers
flows_buffer = []
alerts_buffer = []

if start_button:
    st.experimental_set_query_params(play="1")

    placeholder = flow_feed

    for i in range(100000):
        try:
            # Ideally your inferencer exposes a `/stream` or `/recent` endpoint returning recent scored flows
            resp = requests.get("http://localhost:8000/recent_flows?limit=50", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                df = render_flows(data)
                placeholder.dataframe(df)

                # Extract alerts
                alerts = [f for f in data if f.get("is_alert")]
                if alerts:
                    alerts_df = pd.DataFrame(alerts)
                    alert_table.dataframe(alerts_df)

                    # Show top alert details
                    top = alerts[0]
                    alert_id.json({"id": top.get("flow_id", "-"), "time": top.get("timestamp")})
                    alert_info.json(top)

                    # Fake SHAP (replace with real SHAP values)
                    shap_box.markdown("**Top contributing features**")
                    shap_box.table(
                        pd.DataFrame(top.get("shap", {}), index=[0]).T.rename(columns={0: "value"})
                    )

        except Exception as e:
            st.sidebar.error(f"Polling error: {e}")
            time.sleep(1.0)
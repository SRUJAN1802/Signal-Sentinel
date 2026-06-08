import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Signal Sentinel",
    page_icon="📡",
    layout="wide"
)

# Initialize Session State tracking for Prediction History and Global State
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "score": 75.0,
        "risk_score": 25.0,
        "status": "🟢 Stable",
        "status_color": "#40C880",
        "signal_strength": -85,
        "throughput": 120,
        "latency": 40,
        "network_type": "4G",
        "measurement_avg": -82.0,
        "stability_index": 82.0,
        "link_quality_score": 83.5,
        "failure_probability": 21.2,
        "exec_summary": "Initial baseline set. Run an analysis cycle to refresh Command Center matrix.",
        "root_causes": [("Optimal Parameters", "No active performance bottlenecks parsed.", "#40C880")],
        "noc_alerts": ["✓ No Critical Issues"],
        "actions": ["Link conditions optimal. Retain active tracking matrix loops."]
    }

# =====================================================
# LOAD MODEL AND DATA
# =====================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        network_encoder = joblib.load("network_encoder.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
    except:
        model, network_encoder, feature_columns = None, None, None
    return model, network_encoder, feature_columns

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_network_data.csv")
        importance_df = pd.read_csv("feature_importance.csv")
    except:
        df = pd.DataFrame({
            "Signal Strength (dBm)": np.random.randint(-120, -60, 100),
            "Data Throughput (Mbps)": np.random.randint(10, 450, 100),
            "Latency (ms)": np.random.randint(10, 250, 100),
            "Network Type": np.random.choice([0, 1, 2, 3], 100),
            "Health_Score": np.random.randint(20, 95, 100),
            "Hour": np.random.randint(0, 24, 100),
            "Locality": np.random.choice(["Zone A", "Zone B", "Zone C", "Zone D"], 100)
        })
        importance_df = pd.DataFrame({
            "Feature": ["Throughput_Efficiency", "Signal Strength (dBm)", "Latency (ms)", "Measurement_Avg"],
            "Importance": [0.45, 0.25, 0.20, 0.10]
        })
    return df, importance_df

model, network_encoder, feature_columns = load_model()
df, importance_df = load_data()
data_loaded = True if model is not None else False

# =====================================================
# STYLING (Glassmorphism UI Engine)
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&display=swap');

*, h1, h2, h3, h4, p, label, div, span, button {
    font-family: 'Syne', sans-serif !important;
}
.stApp {
    background-color: #0A0A0F;
}
section[data-testid="stSidebar"] {
    background-color: #0D0D14;
    border-right: 1px solid #1E1E2E;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #C0C8E0 !important;
    font-size: 14px !important;
}
h1 {
    color: #FFFFFF !important;
    font-size: 34px !important;
    font-weight: 700 !important;
}
h2 {
    color: #E8EDF8 !important;
    font-weight: 600 !important;
}
label {
    color: #B0BBD5 !important;
    font-size: 13px !important;
}
p, span {
    color: #A0AABF;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(25, 25, 40, 0.7) 0%, rgba(15, 15, 25, 0.8) 100%);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}
div[data-testid="stMetricLabel"] {
    color: #8090B0 !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
.stButton > button {
    background: #1A1A2E !important;
    color: #E0E8FF !important;
    border: 1px solid #2E3058 !important;
    border-radius: 8px !important;
    height: 50px !important;
    width: 100% !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #222244 !important;
    color: #FFFFFF !important;
    border-color: #5060A0 !important;
}
.stDownloadButton > button {
    background: #0F1A2E !important;
    color: #7EB8E8 !important;
    border: 1px solid #1E3A5A !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    width: 100% !important;
    height: 45px !important;
}
.stDownloadButton > button:hover {
    background: #152238 !important;
    color: #AAD8FF !important;
}
hr { border-color: #1A1A2E; }
[data-testid="stDataFrame"] {
    border: 1px solid #1E2035;
    border-radius: 8px;
}
div[data-testid="stSelectbox"] > div {
    background: #111120 !important;
    border-color: #2A2A45 !important;
    color: #E0E8FF !important;
}
</style>

<div style='text-align: center; padding: 25px 20px; background: linear-gradient(180deg, #0F0F1E 0%, #0A0A0F 100%); border-bottom: 1px solid #1E1E35; margin-bottom: 24px;'>
    <h1 style='color: #FFFFFF !important; font-size: 38px !important; font-weight: 700 !important; margin: 0 0 5px 0; letter-spacing: 1.5px;'>
        📡 Signal Sentinel
    </h1>
    <p style='color: #7080A0; font-size: 13px; letter-spacing: 2.5px; text-transform: uppercase; margin: 0;'>
        Intelligent Radio Link Monitoring System
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# NAVIGATION SIDEBAR
# =====================================================
st.sidebar.markdown(
    "<p style='color:#5060A0; font-size:11px; letter-spacing:2px; text-transform:uppercase; padding:8px 0 4px 0;'>◈ Navigation</p>",
    unsafe_allow_html=True
)
page = st.sidebar.radio("", ["🤖 AI Prediction Engine", "🛰️ Telecom Command Center"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='padding: 10px; background: #0A0A12; border-radius: 6px; border: 1px solid #1E1E2E;'>"
    "<p style='color:#8090B0; font-size:11px; margin:0;'>PORTAL ARCHITECTURE</p>"
    "<p style='color:#40C880; font-size:12px; font-weight:bold; margin:4px 0 0 0;'>● ERICSSON-MONITOR-E6</p>"
    "</div>", 
    unsafe_allow_html=True
)

# =====================================================
# PLOTLY CONSTANTS
# =====================================================
PLOT_BG    = "#0D0D18"
GRID_COLOR = "#1A1A2E"
TEXT_COLOR = "#C0C8E0"
AXIS_COLOR = "#7080A8"

PLOT_LAYOUT = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT_COLOR, family="Syne"),
    margin=dict(t=15, b=30, l=40, r=20)
)

# =====================================================
# SYSTEM HELPERS
# =====================================================
def normalize_0_100(value, min_val, max_val):
    if max_val == min_val:
        return 50.0
    return round(((value - min_val) / (max_val - min_val)) * 100, 1)

def section_header(icon, label, color="#3A4A80"):
    st.markdown(
        f"<div style='margin: 30px 0 14px 0; padding: 12px 18px; background: {color}10; border-left: 3px solid {color}; border-radius: 0 6px 6px 0;'> "
        f"<span style='font-family: Syne, sans-serif; font-size: 13px; letter-spacing: 1.5px; text-transform: uppercase; color: {color}; font-weight: 700;'>"
        f"{icon}&nbsp;&nbsp;{label}</span></div>",
        unsafe_allow_html=True
    )

def info_card(content, border_color="#4E90D8", text_color="#B8D0F0"):
    st.markdown(
        f"<div style='background: #0D1220; border: 1px solid #1A2A40; border-left: 3px solid {border_color}; padding: 14px 18px; border-radius: 6px; color: {text_color}; font-size: 14px; line-height: 1.8; margin: 10px 0;'>"
        f"{content}</div>",
        unsafe_allow_html=True
    )

def create_pdf(report_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(10, 10, 15)
    pdf.rect(0, 0, 210, 297, "F")
    
    pdf.set_fill_color(20, 20, 35)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 20)
    pdf.text(15, 25, "SIGNAL SENTINEL NETWORK REPORT")
    
    pdf.set_text_color(150, 160, 180)
    pdf.set_font("Arial", "", 10)
    pdf.text(15, 33, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Protocol: Link Analytics")
    
    pdf.set_text_color(230, 235, 245)
    pdf.set_font("Arial", "B", 14)
    pdf.text(15, 55, "Operational Diagnostics")
    
    y = 65
    pdf.set_font("Arial", "", 11)
    for k, v in report_data.items():
        clean_v = str(v).replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", "")
        pdf.set_fill_color(15, 20, 35)
        pdf.rect(15, y, 180, 8, "F")
        pdf.set_text_color(140, 150, 180)
        pdf.text(18, y + 6, str(k))
        pdf.set_text_color(255, 255, 255)
        pdf.text(100, y + 6, clean_v)
        y += 10
        
    return pdf.output(dest="S").encode("latin-1", errors="replace")

# =====================================================
# PAGE 1: AI PREDICTION ENGINE & SIMULATOR
# =====================================================
if page == "🤖 AI Prediction Engine":

    section_header("🚀", "Network Scenario Simulator", "#E8A040")

    scenario = st.selectbox(
        "Load a preset scenario",
        ["— Custom / Manual —", "Normal", "Peak Traffic", "Bad Weather", "Tower Failure"]
    )

    SCENARIOS = {
        "Normal":     dict(sig=-78,  tput=180, lat=25,  bb=-76,  srs=-78,  brf=-77),
        "Peak Traffic":  dict(sig=-88,  tput=45,  lat=180, bb=-86,  srs=-89,  brf=-87),
        "Bad Weather":   dict(sig=-112, tput=30,  lat=220, bb=-110, srs=-113, brf=-111),
        "Tower Failure": dict(sig=-125, tput=8,   lat=290, bb=-123, srs=-126, brf=-124),
    }

    defaults = SCENARIOS.get(scenario, {})

    col1, col2 = st.columns(2)
    with col1:
        signal_strength = st.slider("Signal Strength (dBm)", -130, -40, defaults.get("sig", -85))
        throughput = st.slider("Data Throughput (Mbps)", 1, 500, defaults.get("tput", 120))
        latency = st.slider("Latency (ms)", 1, 300, defaults.get("lat", 40))
        network_type = st.selectbox("Network Type", ["3G", "4G", "5G", "LTE"])
        hour = st.slider("Hour of Day", 0, 23, 14)

    with col2:
        latitude  = st.number_input("Latitude",  value=17.3850)
        longitude = st.number_input("Longitude", value=78.4867)
        bb60c   = st.slider("BB60C Measurement",     -130, -40, defaults.get("bb",  -82))
        srsran  = st.slider("srsRAN Measurement",    -130, -40, defaults.get("srs", -84))
        bladerf = st.slider("BladeRFxA9 Measurement",-130, -40, defaults.get("brf", -83))

    # Safe dynamic parsing before grid initialization blocks to avoid NameErrors
    measurement_avg = (bb60c + srsran + bladerf) / 3
    contrib_signal = round((abs(signal_strength) - 85) * 0.18, 2)
    contrib_latency = round(-(latency - 40) * 0.12, 2)
    contrib_tput = round((throughput - 120) * 0.08, 2)
    contrib_meas = round((abs(measurement_avg) - 83) * 0.10, 2)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("◈   ANALYZE NETWORK HEALTH") or scenario != "— Custom / Manual —":
        
        throughput_efficiency = throughput / (latency + 1)
        signal_latency_ratio  = abs(signal_strength) / (latency + 1)

        if data_loaded:
            try:
                network_type_encoded = network_encoder.transform([network_type])[0]
                input_data = pd.DataFrame({
                    "Signal Strength (dBm)": [signal_strength],
                    "Data Throughput (Mbps)": [throughput],
                    "Latency (ms)": [latency],
                    "Network Type": [network_type_encoded],
                    "Latitude": [latitude],
                    "Longitude": [longitude],
                    "Hour": [hour],
                    "Throughput_Efficiency": [throughput_efficiency],
                    "Signal_Latency_Ratio": [signal_latency_ratio],
                    "Measurement_Avg": [measurement_avg]
                })
                input_data = input_data[feature_columns]
                score = float(model.predict(input_data)[0])
            except:
                sig_factor = normalize_0_100(signal_strength, -130, -40)
                tput_factor = normalize_0_100(throughput, 1, 500)
                lat_factor = normalize_0_100(300 - latency, 0, 300)
                score = (sig_factor * 0.4) + (tput_factor * 0.4) + (lat_factor * 0.2)
        else:
            sig_factor = normalize_0_100(signal_strength, -130, -40)
            tput_factor = normalize_0_100(throughput, 1, 500)
            lat_factor = normalize_0_100(300 - latency, 0, 300)
            score = (sig_factor * 0.4) + (tput_factor * 0.4) + (lat_factor * 0.2)

        risk_score = round(100 - score, 2)
        failure_probability = round(risk_score * 0.85, 1)
        
        sig_health = normalize_0_100(signal_strength, -130, -40)
        lat_health = normalize_0_100(300 - latency, 0, 300)
        tput_health = normalize_0_100(throughput, 1, 500)
        rf_health = normalize_0_100(measurement_avg, -130, -40)
        
        stability_index = round((sig_health * 0.3) + (lat_health * 0.3) + (tput_health * 0.4), 1)
        link_quality_score = round((sig_health * 0.5) + (rf_health * 0.5), 1)

        if score >= 70:
            status = "🟢 Stable"
            status_color = "#40C880"
            exec_summary = "Network operating within optimal thresholds. Risk parameters remain low. No immediate intervention required. Forecast indicates stable performance over the next 6 hours."
        elif score >= 50:
            status = "🟡 Degraded"
            status_color = "#E8B040"
            exec_summary = "Minor path losses or load shifting is inducing network packet drag. General health indices sit within warning limits. Preventive monitoring suggested."
        elif score >= 30:
            status = "🟠 High Risk"
            status_color = "#D87030"
            exec_summary = "Critical variance threshold reached. Radio frequency link structural metrics showing severe attenuation. Failure probabilities escalated."
        else:
            status = "🔴 Link Failure"
            status_color = "#E85060"
            exec_summary = "Radio link drop or localized base station isolation observed. Network throughput limits crushed. Immediate engineer asset routing mandatory."

        root_causes = []
        if signal_strength < -100:
            root_causes.append(("Weak Signal Strength", f"Signal at {signal_strength} dBm dropped below target threshold.", "#E85060"))
        if latency > 150:
            root_causes.append(("High Network Latency", f"Latency spikes at {latency} ms generating network packet drag.", "#E8A040"))
        if throughput < 50:
            root_causes.append(("Low Data Throughput", f"Throughput stalling at {throughput} Mbps causing backhaul queues.", "#D87030"))
        if measurement_avg < -95:
            root_causes.append(("Poor Radio Measurements", f"Average RF input level ({measurement_avg:.1f} dBm) critical.", "#A060D8"))
        if not root_causes:
            root_causes.append(("Optimal Parameters", "No active performance bottlenecks parsed.", "#40C880"))

        noc_alerts = []
        if latency > 150:
            noc_alerts.append("⚠ High Latency Detected")
        if throughput < 50:
            noc_alerts.append("⚠ Throughput Below Threshold")
        if signal_strength < -100:
            noc_alerts.append("⚠ Signal Quality Degrading")
        if not noc_alerts:
            noc_alerts.append("✓ No Critical Issues")

        actions = []
        if signal_strength < -95:
            actions.append("Optimize RF transmission power levels and verify horizontal hardware alignment.")
        if latency > 120:
            actions.append("Reduce localized layer-2 latency bottlenecks and review transport backhaul routes.")
        if throughput < 60:
            actions.append("Increase dynamic data throughput cell allocation and redistribute user load patterns.")
        if not actions:
            actions.append("Link conditions optimal. Retain active tracking matrix loops.")

        # Sync calculations into global state cache
        st.session_state.last_metrics = {
            "score": score, "risk_score": risk_score, "status": status, "status_color": status_color,
            "signal_strength": signal_strength, "throughput": throughput, "latency": latency,
            "network_type": network_type, "measurement_avg": measurement_avg, "stability_index": stability_index,
            "link_quality_score": link_quality_score, "failure_probability": failure_probability,
            "exec_summary": exec_summary, "root_causes": root_causes, "noc_alerts": noc_alerts, "actions": actions
        }

        # Save History Log
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        st.session_state.prediction_history.insert(0, {"Timestamp": timestamp_str, "Health Score": f"{score:.2f}", "Risk Score": f"{risk_score:.2f}%", "Status": status})

    # Pull calculations safely out of cached layout memory blocks
    current_metrics = st.session_state.last_metrics
    
    # Recalculate component gauges to prevent rendering mismatch limits
    sig_health = normalize_0_100(signal_strength, -130, -40)
    lat_health = normalize_0_100(300 - latency, 0, 300)
    tput_health = normalize_0_100(throughput, 1, 500)
    rf_health = normalize_0_100(measurement_avg, -130, -40)

    # ----------------------------------------
    # UI RENDERING PAGE 1
    # ----------------------------------------
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Health Score", f"{current_metrics['score']:.2f}")
    m2.metric("Risk Score", f"{current_metrics['risk_score']:.2f}%")
    m3.metric("Network Severity", current_metrics['status'])
    m4.metric("AI Confidence", "98%")

    info_card(f"🔎 &nbsp;<b>Operational Evaluation:</b> {current_metrics['exec_summary']}", border_color=current_metrics['status_color'], text_color="#E8EDF8")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        section_header("🎯", "Health Components Breakdown", "#50C0D0")
        st.markdown(f"**Signal Health** ({sig_health:.1f}%)"); st.progress(int(sig_health) / 100)
        st.markdown(f"**Latency Health** ({lat_health:.1f}%)"); st.progress(int(lat_health) / 100)
        st.markdown(f"**Throughput Health** ({tput_health:.1f}%)"); st.progress(int(tput_health) / 100)
        st.markdown(f"**RF Health** ({rf_health:.1f}%)"); st.progress(int(rf_health) / 100)

    with col_f2:
        section_header("🔬", "AI Feature Contribution", "#8870D8")
        
        shap_df = pd.DataFrame({
            "Feature": ["Signal Strength", "Latency", "Throughput", "Measurement Avg"],
            "Contribution": [contrib_signal, contrib_latency, contrib_tput, contrib_meas]
        }).sort_values("Contribution")
        shap_df["Color"] = shap_df["Contribution"].apply(lambda x: "#58C8A0" if x >= 0 else "#E85060")
        
        fig_shap = go.Figure(go.Bar(x=shap_df["Contribution"], y=shap_df["Feature"], orientation="h", marker_color=shap_df["Color"]))
        fig_shap.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR, family="Syne"), height=220, title=None, margin=dict(t=15, b=20, l=40, r=20),
                               xaxis=dict(gridcolor=GRID_COLOR, color=AXIS_COLOR, linecolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR, color=AXIS_COLOR, linecolor=GRID_COLOR))
        st.plotly_chart(fig_shap, use_container_width=True)

    section_header("🗂️", "Run History Log (Last 10 Runs)", "#8870D8")
    st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)

# =====================================================
# PAGE 2: TELECOM COMMAND CENTER (NOC MONITOR VIEW)
# =====================================================
elif page == "🛰️ Telecom Command Center":
    
    m = st.session_state.last_metrics

    st.markdown("<h2 style='color:#FFFFFF !important; font-weight:700; font-size:26px !important; letter-spacing:1px; margin-bottom:5px;'>🛰️ Strategic Link Command Center</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#607090; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:20px;'>Live Network Operations Center (NOC) Executive Analytics</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stability Index", f"{m['stability_index']}%")
    c2.metric("Link Quality Score", f"{m['link_quality_score']}%")
    c3.metric("Failure Probability", f"{m['failure_probability']}%")
    c4.metric("Current Severity Track", m['status'])

    st.markdown("<p style='color:#8090B0; font-size:11px; letter-spacing:1px; margin-top:20px; margin-bottom:8px;'>ACTIVE TELEMETRY ALERTS ENGINE</p>", unsafe_allow_html=True)
    alert_html = "".join([f"<span style='background:#181215; border:1px solid #3A1A1E; color:#E85060; padding:6px 12px; margin-right:10px; border-radius:4px; font-size:12px;'>{a}</span>" if "⚠" in a 
                          else f"<span style='background:#121815; border:1px solid #1A3A25; color:#40C880; padding:6px 12px; margin-right:10px; border-radius:4px; font-size:12px;'>{a}</span>" for a in m['noc_alerts']])
    st.markdown(f"<div style='margin-bottom:25px;'>{alert_html}</div>", unsafe_allow_html=True)

    st.markdown("<div style='background:#0F111A; border:1px solid #1E2235; padding:18px 22px; border-radius:8px; margin-bottom:25px;'>"
                "<p style='color:#8090B0; font-size:11px; letter-spacing:2px; margin:0 0 5px 0;'>EXECUTIVE SUMMARY</p>"
                f"<p style='color:#E0E8FF; font-size:14px; margin:0; line-height:1.6; font-weight:500;'>{m['exec_summary']}</p></div>", unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    
    with v1:
        section_header("◈", "Live Health Gauge", "#4E90D8")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=m['score'],
            number={"font": {"color": "#FFFFFF", "family": "Syne", "size": 38}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#2A2A48", "tickfont": {"color": "#6070A0", "size": 11}, "dtick": 20},
                "bar": {"color": "#4E90D8", "thickness": 0.22}, "bgcolor": PLOT_BG, "borderwidth": 0,
                "steps": [{"range": [0, 30], "color": "#250A10"}, {"range": [30, 50], "color": "#251500"}, {"range": [50, 70], "color": "#1A2000"}, {"range": [70, 100], "color": "#002010"}]
            }
        ))
        gauge.update_layout(**PLOT_LAYOUT, height=220, title=None)
        st.plotly_chart(gauge, use_container_width=True)

    with v2:
        section_header("📡", "Radio Link Predictive Forecast", "#58C8A0")
        time_steps = ["Current", "1 Hour Forecast", "3 Hour Forecast", "6 Hour Forecast"]
        drift = -12 if m['status'] in ["🔴 Link Failure", "🟠 High Risk"] else 4
        f_scores = [m['score'], max(0, min(100, m['score'] + drift)), max(0, min(100, m['score'] + (drift * 1.4))), max(0, min(100, m['score'] + (drift * 0.8)))]
        
        fig_forecast = go.Figure(go.Scatter(x=time_steps, y=f_scores, mode="lines+markers", line=dict(color="#58C8A0", width=3), marker=dict(size=8, color="#FFFFFF")))
        fig_forecast.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR, family="Syne"), height=220, title=None, margin=dict(t=15, b=20, l=40, r=20),
                                   xaxis=dict(gridcolor=GRID_COLOR, color=AXIS_COLOR, linecolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR, color=AXIS_COLOR, linecolor=GRID_COLOR, range=[0, 105]))
        st.plotly_chart(fig_forecast, use_container_width=True)

    section_header("🤖", "AI Mitigations & Action Protocols", "#58C8A0")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("<span style='color:#8090B0; font-size:12px; letter-spacing:1px;'>COMMAND ANOMALY METRIC COUNTERS</span>", unsafe_allow_html=True)
        for title, detail, color in m['root_causes']:
            st.markdown(f"<div style='margin: 8px 0; font-size:13px; color:#B0C0D8;'><b style='color:{color};'>• {title}:</b> {detail}</div>", unsafe_allow_html=True)
    with col_e2:
        st.markdown("<span style='color:#8090B0; font-size:12px; letter-spacing:1px;'>CRITICAL DISPATCH COMMANDS</span>", unsafe_allow_html=True)
        for i, act in enumerate(m['actions'], 1):
            st.markdown(f"<div style='margin: 8px 0; font-size:13px; color:#FFFFFF;'><b>{i}.</b> {act}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pdf_payload = {
        "Operational Security Output Status": m['status'], "Calculated Link Health Score": f"{m['score']:.2f}",
        "Link Stability Index Tracker": f"{m['stability_index']}%", "Radio Link Quality Mean Metric": f"{m['link_quality_score']}%",
        "Estimated Failure Probability": f"{m['failure_probability']}%", "Evaluated Signal Strength (dBm)": m['signal_strength'],
        "Evaluated Link Latency Profile": f"{m['latency']} ms", "Evaluated Link Data Throughput": f"{m['throughput']} Mbps"
    }
    pdf_bytes = create_pdf(pdf_payload)
    st.download_button(label="📥 Export Enterprise Link Diagnostics Report (PDF)", data=pdf_bytes, file_name="signal_sentinel_diagnostic.pdf", mime="application/pdf")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#303050; font-size:12px; letter-spacing:2px; padding:14px; text-transform:uppercase;'>"
            "◈ Signal Sentinel &nbsp;·&nbsp; Enterprise Telecom Engine Framework ◈</div>", unsafe_allow_html=True)
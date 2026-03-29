"""
=============================================================
 dashboard.py — Drowsy Driver Detection Results Dashboard
 Run with: streamlit run dashboard.py
=============================================================
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from pathlib import Path
 
# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Drowsy Driver Detection — Results Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# =============================================================
# CUSTOM CSS
# =============================================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4266;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ccd6f6;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    .stMetric { background: #1e2130; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)
 
# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.image("https://img.icons8.com/fluency/96/car.png", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "📊 Project Overview",
    "🧠 Model Training",
    "✅ Test Results",
    "📹 Session Logs",
    "ℹ️ About"
])
 
st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Drowsy Driver Detection")
st.sidebar.markdown("**Model:** ResNet-18 (Fine-tuned)")
st.sidebar.markdown("**Dataset:** DDD Benchmark")
st.sidebar.markdown("**Team:** Medha Tyagi, Param Varsha")
 
# =============================================================
# DATA
# =============================================================
epochs     = list(range(1, 16))
train_acc  = [99.38,99.75,99.90,99.93,99.99,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0]
val_acc    = [99.98,99.98,99.98,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0]
train_loss = [0.0936,0.0134,0.0055,0.0082,0.0007,0.0001,0.0065,0.0000,0.0002,0.0002,0.0001,0.0001,0.0000,0.0000,0.0000]
val_loss   = [0.0030,0.0035,0.0000,0.0007,0.0000,0.0000,0.0000,0.0000,0.0001,0.0001,0.0000,0.0000,0.0000,0.0000,0.0000]
 
cm = np.array([[703, 0], [0, 393]])
 
# =============================================================
# PAGE 1 — OVERVIEW
# =============================================================
if page == "📊 Project Overview":
    st.title("🚗 Drowsy Driver Detection System")
    st.markdown("**Real-Time Multi-Factor Behavioral Fusion using Deep Learning**")
    st.markdown("---")
 
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("100.00%", "Accuracy"),
        ("100.00%", "F1 Score"),
        ("1.0000",  "ROC-AUC"),
        ("22.6 min","Training Time"),
        ("25-28",   "FPS (Real-Time)"),
    ]
    for col, (val, label) in zip([col1,col2,col3,col4,col5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{val}</p>
                <p class="metric-label">{label}</p>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
    col_l, col_r = st.columns(2)
 
    with col_l:
        st.markdown('<p class="section-header">System Architecture</p>', unsafe_allow_html=True)
        components = {
            "Component": ["Eye Closure (EAR)", "Head Tilt", "Hand Stability", "DAI Fusion", "Alert System"],
            "Method": ["MediaPipe + ResNet-18 CNN", "MediaPipe FaceMesh Pitch", "MediaPipe Wrist Tracking", "Weighted Sum (0.5/0.3/0.2)", "3-Level Graduated Alerts"],
            "Weight": ["0.5", "0.3", "0.2", "—", "—"]
        }
        st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)
 
    with col_r:
        st.markdown('<p class="section-header">Dataset Statistics</p>', unsafe_allow_html=True)
        dataset = {
            "Category": ["Raw Drowsy", "Raw Non-Drowsy", "Duplicates Removed", "Clean Drowsy", "Clean Non-Drowsy", "Total Clean"],
            "Count": [22348, 19445, 15423, 6925, 4022, 10947]
        }
        df_data = pd.DataFrame(dataset)
        st.dataframe(df_data, use_container_width=True, hide_index=True)
 
        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0e1117")
        colors = ["#00d4ff", "#ff6b6b"]
        ax.pie([6925, 4022], labels=["Drowsy\n6,925", "Non-Drowsy\n4,022"],
               colors=colors, autopct="%1.1f%%", startangle=90,
               textprops={"color": "white", "fontsize": 10})
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig)
        plt.close()
 
    st.markdown("---")
    st.markdown('<p class="section-header">Alert Threshold System</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🟡 **MILD ALERT**\nDAI: 0.3 — 0.6\nSoft audio tone\nDriver awareness nudge")
    with col2:
        st.warning("🟠 **WARNING ALERT**\nDAI: 0.6 — 0.8\nLouder audio tone\nUrgent notification")
    with col3:
        st.error("🔴 **CRITICAL ALERT**\nDAI: > 0.8\nAudio + Visual flash\nImmediate response required")
 
# =============================================================
# PAGE 2 — MODEL TRAINING
# =============================================================
elif page == "🧠 Model Training":
    st.title("🧠 Model Training Analysis")
    st.markdown("---")
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Architecture", "ResNet-18")
    col2.metric("Total Params", "11,308,354")
    col3.metric("Trainable Params", "10,625,282")
    col4.metric("Training Time", "22.6 min")
 
    st.markdown("---")
    col_l, col_r = st.columns(2)
 
    with col_l:
        st.markdown('<p class="section-header">Accuracy per Epoch</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
        ax.plot(epochs, train_acc, "o-", color="#00d4ff", linewidth=2, markersize=5, label="Train")
        ax.plot(epochs, val_acc,   "o-", color="#ff6b6b", linewidth=2, markersize=5, label="Validation")
        ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("Accuracy (%)", color="white")
        ax.set_title("Training vs Validation Accuracy", color="white", fontsize=13)
        ax.set_ylim([96, 101])
        ax.tick_params(colors="white"); ax.spines[:].set_color("#3d4266")
        ax.set_facecolor("#1e2130"); fig.patch.set_facecolor("#0e1117")
        ax.legend(facecolor="#1e2130", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig); plt.close()
 
    with col_r:
        st.markdown('<p class="section-header">Loss per Epoch</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0e1117")
        ax.plot(epochs, train_loss, "o-", color="#00d4ff", linewidth=2, markersize=5, label="Train")
        ax.plot(epochs, val_loss,   "o-", color="#ff6b6b", linewidth=2, markersize=5, label="Validation")
        ax.set_xlabel("Epoch", color="white"); ax.set_ylabel("Loss", color="white")
        ax.set_title("Training vs Validation Loss", color="white", fontsize=13)
        ax.tick_params(colors="white"); ax.spines[:].set_color("#3d4266")
        ax.set_facecolor("#1e2130"); fig.patch.set_facecolor("#0e1117")
        ax.legend(facecolor="#1e2130", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig); plt.close()
 
    st.markdown("---")
    st.markdown('<p class="section-header">Epoch-by-Epoch Results</p>', unsafe_allow_html=True)
    df_epochs = pd.DataFrame({
        "Epoch": epochs,
        "Train Loss": train_loss,
        "Train Acc (%)": train_acc,
        "Val Loss": val_loss,
        "Val Acc (%)": val_acc
    })
    st.dataframe(df_epochs, use_container_width=True, hide_index=True)
 
# =============================================================
# PAGE 3 — TEST RESULTS
# =============================================================
elif page == "✅ Test Results":
    st.title("✅ Test Set Evaluation")
    st.markdown("---")
 
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy",  "100.00%")
    col2.metric("Precision", "100.00%")
    col3.metric("Recall",    "100.00%")
    col4.metric("F1 Score",  "100.00%")
    col5.metric("ROC-AUC",   "1.0000")
 
    st.markdown("---")
    col_l, col_r = st.columns(2)
 
    with col_l:
        st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0e1117")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Drowsy","Non-Drowsy"],
                    yticklabels=["Drowsy","Non-Drowsy"],
                    annot_kws={"size":18, "color":"white"},
                    ax=ax)
        ax.set_title("Confusion Matrix — Test Set", color="white", fontsize=13)
        ax.set_ylabel("True Label", color="white")
        ax.set_xlabel("Predicted Label", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig); plt.close()
 
    with col_r:
        st.markdown('<p class="section-header">ROC Curve</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0e1117")
        ax.plot([0,0,1],[0,1,1], color="#00d4ff", lw=2, label="Model (AUC = 1.0000)")
        ax.plot([0,1],[0,1], "r--", lw=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", color="white")
        ax.set_ylabel("True Positive Rate", color="white")
        ax.set_title("ROC Curve — Drowsy vs Non-Drowsy", color="white", fontsize=13)
        ax.tick_params(colors="white"); ax.spines[:].set_color("#3d4266")
        ax.set_facecolor("#1e2130"); fig.patch.set_facecolor("#0e1117")
        ax.legend(facecolor="#1e2130", labelcolor="white")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig); plt.close()
 
    st.markdown("---")
    st.markdown('<p class="section-header">Multi-Factor vs Single-Channel Detection</p>', unsafe_allow_html=True)
    comparison = {
        "Condition": ["Normal Lighting", "Low-Light", "Sunglasses (Occlusion)", "Head Upright", "Hands Off Wheel"],
        "Eye Only":   ["✅", "⚠️ Degraded", "❌", "✅", "✅"],
        "Head Only":  ["✅", "✅", "✅", "❌", "✅"],
        "Hand Only":  ["✅", "✅", "✅", "✅", "❌"],
        "DAI Fusion": ["✅", "✅", "✅", "✅", "✅"],
    }
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
 
# =============================================================
# PAGE 4 — SESSION LOGS
# =============================================================
elif page == "📹 Session Logs":
    st.title("📹 Session Log Viewer")
    st.markdown("View real-time detection logs saved during webcam sessions.")
    st.markdown("---")
 
    log_dir = "./logs"
    if os.path.exists(log_dir):
        log_files = sorted(Path(log_dir).glob("session_*.csv"), reverse=True)
        if log_files:
            selected = st.selectbox("Select session log:", [f.name for f in log_files])
            df_log = pd.read_csv(os.path.join(log_dir, selected))
            st.markdown(f"**Total alerts in session: {len(df_log)}**")
 
            col1, col2, col3 = st.columns(3)
            col1.metric("Mild Alerts",     len(df_log[df_log["alert_level"]=="MILD"]))
            col2.metric("Warning Alerts",  len(df_log[df_log["alert_level"]=="WARNING"]))
            col3.metric("Critical Alerts", len(df_log[df_log["alert_level"]=="CRITICAL"]))
 
            st.markdown("---")
            st.dataframe(df_log, use_container_width=True, hide_index=True)
 
            if len(df_log) > 0:
                st.markdown("**DAI Values During Session**")
                fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
                ax.plot(df_log["dai"].values, color="#00d4ff", linewidth=1.5)
                ax.axhline(y=0.3, color="yellow", linestyle="--", alpha=0.5, label="Mild (0.3)")
                ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.5, label="Warning (0.6)")
                ax.axhline(y=0.8, color="red",    linestyle="--", alpha=0.5, label="Critical (0.8)")
                ax.set_xlabel("Alert Event", color="white")
                ax.set_ylabel("DAI Value", color="white")
                ax.set_title("DAI Over Session", color="white")
                ax.set_facecolor("#1e2130"); fig.patch.set_facecolor("#0e1117")
                ax.tick_params(colors="white"); ax.spines[:].set_color("#3d4266")
                ax.legend(facecolor="#1e2130", labelcolor="white")
                ax.grid(True, alpha=0.2)
                st.pyplot(fig); plt.close()
        else:
            st.info("No session logs found. Run detect.py first to generate logs.")
    else:
        st.info("No logs directory found. Run detect.py first to generate session logs.")
 
# =============================================================
# PAGE 5 — ABOUT
# =============================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
 
    st.markdown("""
    ### Real-Time Drowsy Driver Detection Using Multi-Factor Behavioral Fusion and Deep Learning
 
    **Authors:** Medha Tyagi, Param Varsha
    **Institution:** SRM Institute of Science and Technology, Kattankulathur
 
    ---
 
    ### How It Works
 
    The system monitors three behavioral signals simultaneously:
 
    1. **Eye Closure (EAR)** — Uses MediaPipe FaceMesh to compute Eye Aspect Ratio.
       A fine-tuned ResNet-18 CNN provides additional confidence scoring.
 
    2. **Head Tilt** — Estimates forward pitch angle from facial landmarks.
       Angles beyond 20° indicate microsleep or fatigue.
 
    3. **Hand Stability** — Tracks wrist movement via MediaPipe Hands.
       Reduced movement indicates relaxed, fatigued grip.
 
    These three signals are fused into the **Driver Attention Index (DAI)**:
 
    > DAI = 0.5 × f_eye + 0.3 × f_head + 0.2 × f_hand
 
    The DAI triggers graduated alerts at three severity levels,
    eliminating alarm fatigue from binary threshold systems.
 
    ---
 
    ### Key Results
    - **100% accuracy** on held-out test set (DDD benchmark)
    - **25–28 FPS** real-time performance on consumer hardware
    - **Occlusion resilient** — system continues detecting via head/hand when eyes are blocked
    - **Personalized calibration** per driver session
 
    ---
 
    ### GitHub Repository
    [github.com/paramvarsha12/Drowsy-Driver-Detection-Multi-Factor-Behavioral-Fusion](https://github.com/paramvarsha12/Drowsy-Driver-Detection-Multi-Factor-Behavioral-Fusion)
    """)
 
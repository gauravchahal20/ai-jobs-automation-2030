import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Risk Predictor 2030",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS (dark futuristic theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root reset ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0b0f !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] * {
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1a1d26 !important;
    border: 1px solid rgba(255,255,255,0.13) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
    padding-top: 1.5rem !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #00e5ff !important;
}
[data-testid="stSlider"] > div > div > div {
    background: #1a1d26 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,229,255,0.15)) !important;
    color: #00e5ff !important;
    border: 1px solid rgba(0,229,255,0.35) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,229,255,0.15) !important;
    border-color: #00e5ff !important;
    transform: none !important;
}

/* ── Metric boxes ── */
[data-testid="metric-container"] {
    background: #111318 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 11px !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00e5ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.4rem !important;
}

/* ── Headers / text ── */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #e8eaf0 !important;
    letter-spacing: -0.02em !important;
}
p, li, span, label {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 0.8rem 0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111318 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

/* ── Alert/info boxes ── */
[data-testid="stAlert"] {
    background: #111318 !important;
    border-radius: 10px !important;
    border-left: 3px solid #00e5ff !important;
}

/* ── Plotly chart background ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .svg-container {
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

/* ── Main block padding ── */
.block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* 🚫 Hide sidebar hover arrow */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

button[title="Collapse sidebar"],
button[title="Expand sidebar"] {
    display: none !important;
}

/* Custom card styling */
.custom-card {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 26px;
    margin-bottom: 1.4rem;
}

.insight-box {
    background: #111318;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 13px 16px;
    font-size: 13px;
    color: #c9cdd6;
    line-height: 1.65;
    margin-bottom: 1rem;
}

.pill-tag {
    display: inline-block;
    background: #1a1d26;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #a0a8b8;
    margin: 3px 4px 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "columns.pkl"
JOB_PATH = BASE_DIR / "data" / "processed" / "job_profiles.csv"

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load model, columns, and job data"""
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    job_df = pd.read_csv(JOB_PATH)
    return model, columns, job_df

try:
    model, columns, job_df = load_assets()
    assets_ok = True
except Exception as e:
    assets_ok = False
    st.markdown(f"""
    <div class="insight-box" style="border-left:3px solid #ef4444;color:#fca5a5;">
        <strong>⚠️ Error loading assets:</strong> {e}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if assets_ok:
    if 'selected_job' not in st.session_state:
        st.session_state.selected_job = job_df.iloc[0]["job_role"]
    if 'slider_values' not in st.session_state:
        st.session_state.slider_values = None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:16px;
                border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:4px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#7c3aed,#00e5ff);
                    border-radius:8px;flex-shrink:0;"></div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
                    color:#00e5ff;letter-spacing:0.08em;text-transform:uppercase;line-height:1.3;">
            AI Risk<br>Predictor 2030
        </div>
    </div>
    """, unsafe_allow_html=True)

    if assets_ok:
        # Job selector
        st.markdown("""
        <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
                  letter-spacing:0.12em;color:#6b7280;text-transform:uppercase;margin-bottom:6px;">
        Job Role</p>
        """, unsafe_allow_html=True)

        job = st.selectbox(
            "Select a job role",
            job_df["job_role"].unique(),
            index=list(job_df["job_role"].unique()).index(st.session_state.selected_job),
            label_visibility="collapsed",
            key="job_selector",
        )
        st.session_state.selected_job = job
        selected_job = job_df[job_df["job_role"] == job].iloc[0]

        st.markdown("<hr>", unsafe_allow_html=True)

        use_defaults = st.session_state.slider_values is None
        sv = st.session_state.slider_values or {}

        # Get default values from selected job
        d_exp = int(selected_job.get("experience_required_years", 5))
        d_creativity = float(selected_job.get("creativity_requirement", 0.5))
        d_analytical = float(selected_job.get("analytical_complexity", 0.5))
        d_ai_future = float(selected_job.get("ai_dependency_future", 0.5))
        d_physical = float(selected_job.get("physical_labor_level", 0.5))
        d_social = float(selected_job.get("social_interaction_level", 0.5))
        d_salary = int(selected_job.get("avg_salary_usd", 50000))
        d_growth = float(selected_job.get("job_growth_rate", 0.0))
        d_demand = float(selected_job.get("job_demand_index", 0.5))

        # ── Skills & Experience ──
        st.markdown("""
        <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
                  letter-spacing:0.12em;color:#6b7280;text-transform:uppercase;margin-bottom:2px;">
        Skills & Experience</p>
        """, unsafe_allow_html=True)

        exp = st.slider(
            "Experience (years)", 0, 20,
            value=d_exp if use_defaults else sv.get("exp", d_exp),
            key="exp"
        )
        creativity = st.slider(
            "Creativity requirement", 0.0, 1.0,
            value=d_creativity if use_defaults else sv.get("creativity", d_creativity),
            key="creativity"
        )
        analytical = st.slider(
            "Analytical complexity", 0.0, 1.0,
            value=d_analytical if use_defaults else sv.get("analytical", d_analytical),
            key="analytical"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Work Environment ──
        st.markdown("""
        <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
                  letter-spacing:0.12em;color:#6b7280;text-transform:uppercase;margin-bottom:2px;">
        Work Environment</p>
        """, unsafe_allow_html=True)

        ai_future = st.slider(
            "AI dependency (future)", 0.0, 1.0,
            value=d_ai_future if use_defaults else sv.get("ai_future", d_ai_future),
            key="ai_future"
        )
        physical = st.slider(
            "Physical labor", 0.0, 1.0,
            value=d_physical if use_defaults else sv.get("physical", d_physical),
            key="physical"
        )
        social = st.slider(
            "Social interaction", 0.0, 1.0,
            value=d_social if use_defaults else sv.get("social", d_social),
            key="social"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Market Factors ──
        st.markdown("""
        <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
                  letter-spacing:0.12em;color:#6b7280;text-transform:uppercase;margin-bottom:2px;">
        Market Factors</p>
        """, unsafe_allow_html=True)

        salary = st.slider(
            "Salary (USD)", 10000, 200000,
            value=d_salary if use_defaults else sv.get("salary", d_salary),
            step=1000, key="salary"
        )
        job_growth = st.slider(
            "Job growth rate", -0.5, 0.5,
            value=d_growth if use_defaults else sv.get("growth", d_growth),
            step=0.01, key="growth"
        )
        demand = st.slider(
            "Demand index", 0.0, 1.0,
            value=d_demand if use_defaults else sv.get("demand", d_demand),
            key="demand"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Reset button
        if st.button("↺ Reset to defaults", use_container_width=True):
            st.session_state.slider_values = None
            st.rerun()

        # Store slider values
        st.session_state.slider_values = {
            "exp": exp, "creativity": creativity, "analytical": analytical,
            "ai_future": ai_future, "physical": physical, "social": social,
            "salary": salary, "growth": job_growth, "demand": demand,
        }

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if not assets_ok:
    st.stop()

# ── FEATURE ENGINEERING ──
data = {
    # Base features
    "experience_required_years": exp,
    "creativity_requirement": creativity,
    "analytical_complexity": analytical,
    "ai_dependency_future": ai_future,
    "physical_labor_level": physical,
    "social_interaction_level": social,
    "job_growth_rate": job_growth,
    "job_demand_index": demand,
    
    # Additional core features
    "task_repetition_level": 1 - creativity,
    "percent_tasks_automatable": ai_future,
    "communication_requirement": social,
    "avg_salary_usd": salary,
}

# Derived features
data["experience_inverse"] = 1 / (exp + 1)
data["salary_scaled"] = salary / 100000
data["job_stability"] = (job_growth + demand) / 2
data["human_resistance"] = 0.4 * creativity + 0.3 * social + 0.3 * (1 - ai_future)
data["skill_index"] = analytical
data["ai_readiness"] = ai_future
data["automation_pressure"] = 0.4 * (1 - creativity) + 0.4 * ai_future + 0.2 * (1 - analytical)

# Prepare input for model
input_df = pd.DataFrame([data])
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

# Make prediction
prediction = model.predict(input_df)[0]
risk_pct = prediction * 100

# ─────────────────────────────────────────────
# EXPLAINABILITY (WHY THIS RISK)
# ─────────────────────────────────────────────
reasons = []

if ai_future > 0.7:
    reasons.append("High AI dependency increases automation risk")

if data["task_repetition_level"] > 0.7:
    reasons.append("Highly repetitive tasks are easy to automate")

if creativity < 0.4:
    reasons.append("Low creativity makes job easier for AI")

if social < 0.4:
    reasons.append("Low human interaction reduces need for humans")

if analytical < 0.4:
    reasons.append("Low analytical complexity increases automation chances")

if salary < 40000:
    reasons.append("Lower salary roles are more prone to automation")

if data["human_resistance"] > 0.7:
    reasons.append("High human interaction reduces automation risk")
    
# Determine risk level
if risk_pct < 33:
    risk_level = "LOW RISK"
    risk_color = "#10b981"
    risk_bg = "rgba(16,185,129,0.1)"
    risk_border = "rgba(16,185,129,0.3)"
    insight = "High creativity and social interaction create strong human-only barriers to automation."
elif risk_pct < 66:
    risk_level = "MEDIUM RISK"
    risk_color = "#f59e0b"
    risk_bg = "rgba(245,158,11,0.1)"
    risk_border = "rgba(245,158,11,0.3)"
    insight = "Some task automation is likely. Focus on developing uniquely human skills to stay relevant."
else:
    risk_level = "HIGH RISK"
    risk_color = "#ef4444"
    risk_bg = "rgba(239,68,68,0.1)"
    risk_border = "rgba(239,68,68,0.3)"
    insight = "Significant automation risk. AI tools will likely handle core tasks. Consider upskilling or pivoting."

# ── JOB HEADER ──
st.markdown(f"""
<div class="custom-card" style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
    <div>
        <div style="font-family:'Space Mono',monospace;font-size:24px;font-weight:700;
                    color:#e8eaf0;letter-spacing:-0.02em;line-height:1.2;">{job}</div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:#6b7280;
                    letter-spacing:0.07em;text-transform:uppercase;margin-top:6px;">
            2030 Automation Risk Analysis
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:8px;padding:8px 16px;border-radius:20px;
                background:{risk_bg};border:1px solid {risk_border};
                font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
                letter-spacing:0.1em;color:{risk_color};white-space:nowrap;flex-shrink:0;">
        <div style="width:8px;height:8px;border-radius:50%;background:{risk_color};"></div>
        {risk_level}
    </div>
</div>
""", unsafe_allow_html=True)

# ── TWO-COLUMN LAYOUT ──
col_left, col_right = st.columns([1, 1], gap="medium")

# ── LEFT COLUMN: GAUGE & METRICS ──
with col_left:
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
              letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:4px;">
    Automation Risk Score</p>
    """, unsafe_allow_html=True)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={
            'suffix': '%',
            'font': {'size': 44, 'family': 'Space Mono', 'color': risk_color},
        },
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 0,
                'tickcolor': "rgba(255,255,255,0.15)",
                'tickfont': {'color': 'rgba(150,160,180,0.6)', 'size': 10, 'family': 'Space Mono'},
            },
            'bar': {'color': risk_color, 'thickness': 0.18},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': 'rgba(16,185,129,0.18)'},
                {'range': [33, 66], 'color': 'rgba(245,158,11,0.18)'},
                {'range': [66, 100], 'color': 'rgba(239,68,68,0.18)'},
            ],
            'threshold': {
                'line': {'color': risk_color, 'width': 3},
                'thickness': 0.85,
                'value': risk_pct,
            },
        },
    ))

    fig_gauge.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Space Mono', 'color': '#e8eaf0'},
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    # Insight box
    st.markdown(f"""
    <div class="insight-box" style="border-left:3px solid {risk_color};">
        {insight}
    </div>
    """, unsafe_allow_html=True)
     
    # ─────────────────────────────────────────────
    # SHOW WHY THIS RISK
    # ─────────────────────────────────────────────
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
        letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:4px;">
    Why This Risk</p>
    """, unsafe_allow_html=True)

    if reasons:
      for r in reasons:
        st.markdown(f'<div class="insight-box">• {r}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-box">• Balanced job profile with mixed signals</div>', unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
              letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:4px;">
    Key Metrics</p>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Job Stability", f"{data['job_stability']:.2f}")
    with m2:
        st.metric("Human Resistance", f"{data['human_resistance']:.2f}")
    with m3:
        st.metric("AI Readiness", f"{data['ai_readiness']:.2f}")

# ── RIGHT COLUMN: RADAR & SIMILAR JOBS ──
with col_right:
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
              letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:4px;">
    Feature Profile</p>
    """, unsafe_allow_html=True)

    categories = ['Creativity', 'Analytical', 'AI Dependency', 'Physical', 'Social']
    current_vals = [creativity, analytical, ai_future, physical, social]
    baseline_vals = [d_creativity, d_analytical, d_ai_future, d_physical, d_social]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=current_vals + [current_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Current Settings',
        line=dict(color='#00e5ff', width=2),
        fillcolor='rgba(0,229,255,0.10)',
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=baseline_vals + [baseline_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Job Default',
        line=dict(color='rgba(255,255,255,0.25)', width=1.5, dash='dot'),
        fillcolor='rgba(255,255,255,0.04)',
    ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=9, color='rgba(150,160,180,0.5)', family='Space Mono'),
                gridcolor='rgba(255,255,255,0.07)',
                linecolor='rgba(255,255,255,0.07)',
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='rgba(200,210,230,0.8)', family='DM Sans'),
                gridcolor='rgba(255,255,255,0.07)',
                linecolor='rgba(255,255,255,0.07)',
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=11, color='rgba(200,210,230,0.7)', family='DM Sans'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.07)',
            borderwidth=1,
        ),
        height=320,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # Similar roles
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
              letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-bottom:6px;">
    Similar Roles</p>
    """, unsafe_allow_html=True)

    # Find similar jobs based on creativity and analytical scores
    if 'creativity_requirement' in job_df.columns and 'analytical_complexity' in job_df.columns:
        similar = job_df[
            (job_df['creativity_requirement'].between(creativity - 0.2, creativity + 0.2)) &
            (job_df['analytical_complexity'].between(analytical - 0.2, analytical + 0.2))
        ]["job_role"].head(6).tolist()
        similar = [j for j in similar if j != job][:5]
    else:
        similar = []

    if similar:
        pills_html = "".join([
            f'<span class="pill-tag">{j}</span>'
            for j in similar
        ])
        st.markdown(f'<div style="line-height:2;">{pills_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="font-size:12px;color:#6b7280;">No similar roles found — try adjusting sliders.</p>',
            unsafe_allow_html=True
        )
    # ─────────────────────────────────────────────
    # SAFER JOB RECOMMENDATIONS (ADD HERE 👇)
    # ─────────────────────────────────────────────
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
        letter-spacing:0.14em;color:#6b7280;text-transform:uppercase;margin-top:10px;">
    Safer Alternatives</p>
    """, unsafe_allow_html=True)

    try:
        safer_jobs = job_df[
            (job_df["ai_dependency_future"] < ai_future) &
            (job_df["creativity_requirement"] >= creativity)
        ]["job_role"].head(5).tolist()
    
        # Remove current job
        safer_jobs = [j for j in safer_jobs if j != job]
    
        if safer_jobs:
            pills_html = "".join([
                f'<span class="pill-tag">{j}</span>'
                for j in safer_jobs
            ])
            st.markdown(f'<div style="line-height:2;">{pills_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="font-size:12px;color:#6b7280;">No safer roles found.</p>',
                unsafe_allow_html=True
            )
    except:
        st.markdown(
            '<p style="font-size:12px;color:#6b7280;">Recommendations unavailable.</p>',
            unsafe_allow_html=True
        )   
        

# ── FOOTER ──
st.markdown("<hr style='margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:10px;
            color:#4b5563;letter-spacing:0.08em;text-transform:uppercase;padding-bottom:1rem;">
    🤖 AI Job Automation Risk Predictor 2030 · Model trained on synthetic job market data · Results are trend-based estimates
</div>
""", unsafe_allow_html=True)
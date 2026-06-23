import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & ENTERPRISE RISK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Risk Command Center", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        background-color: #0f172a; 
        color: #e2e8f0; 
    }
    .stApp { 
        background-color: #0f172a; 
    }
    
    /* Structured Executive Cards */
    .metric-card { 
        background: #1e293b; 
        border: 1px solid #334155; 
        border-radius: 6px; 
        padding: 1.5rem; 
        margin-bottom: 1rem; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); 
    }
    .metric-title { 
        color: #94a3b8; 
        font-size: 0.75rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.05em; 
    }
    .metric-value { 
        color: #f8fafc; 
        font-size: 2.2rem; 
        font-weight: 700; 
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.4rem; 
        letter-spacing: -0.02em; 
    }
    .metric-subtext {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    
    /* System Decision Badges */
    .decision-badge { 
        padding: 0.5rem 1rem; 
        border-radius: 4px; 
        font-size: 0.8rem; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
        display: inline-block;
        text-align: center;
        width: 100%;
    }
    .badge-critical { 
        background-color: rgba(239, 68, 68, 0.1); 
        color: #ef4444; 
        border: 1px solid rgba(239, 68, 68, 0.3); 
    }
    .badge-suspicious { 
        background-color: rgba(250, 204, 21, 0.1); 
        color: #facc15; 
        border: 1px solid rgba(250, 204, 21, 0.3); 
    }
    .badge-verified { 
        background-color: rgba(16, 185, 129, 0.1); 
        color: #10b981; 
        border: 1px solid rgba(16, 185, 129, 0.3); 
    }
    
    /* Tables Overrides */
    .stDataFrame { 
        background-color: #1e293b; 
        border-radius: 6px; 
    }
    [data-testid="stTable"] { 
        background-color: #1e293b; 
        border-radius: 6px; 
    }
    [data-testid="stTable"] th { 
        background-color: #0f172a !important; 
        color: #94a3b8 !important; 
        text-transform: uppercase; 
        font-size: 0.75rem; 
        letter-spacing: 0.05em; 
        border-bottom: 1px solid #334155 !important; 
    }
    [data-testid="stTable"] td { 
        color: #e2e8f0 !important; 
        font-size: 0.85rem; 
        border-bottom: 1px solid #1e293b !important; 
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA ACQUISITION & CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_scored_data():
    try:
        data = pd.read_csv("scored_transactions.csv")
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        return data
    except FileNotFoundError:
        return None

df = load_scored_data()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
c_header1, c_header2 = st.columns([3, 1])
with c_header1:
    st.markdown("<h1 style='color:#f8fafc; font-size:1.8rem; margin-bottom:0;'>FRAUD RISK OPERATIONS COMMAND CENTER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.05em;'>Machine Learning Transaction Evaluation & Risk Allocation Ledger</p>", unsafe_allow_html=True)
with c_header2:
    st.markdown(
        "<div style='text-align:right; margin-top:1rem;'>"
        "<span class='decision-badge badge-verified' style='width:auto;'>SYSTEM STATUS: SECURE</span>"
        "</div>", 
        unsafe_allow_html=True
    )

st.markdown("<hr style='border-color:#334155; margin-top:0.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

if df is None:
    st.error("SYSTEM ERROR: scored_transactions.csv not found. Please run your machine learning pipeline (engine.py) first to generate the outputs.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE PORTFOLIO STATS (CROSS-REFERENCING IMAGE_81CB01.PNG METRICS)
# ─────────────────────────────────────────────────────────────────────────────
# Total volumes inspected
total_inspected = len(df)

# True Positives are those marked as actual fraud which fell within the alert threshold (Fraud_Probability >= 0.3)
true_positives = df[(df['Is_Fraud'] == 1) & (df['Fraud_Probability'] >= 0.3)]
false_positives = df[(df['Is_Fraud'] == 0) & (df['Fraud_Probability'] >= 0.3)]
false_negatives = df[(df['Is_Fraud'] == 1) & (df['Fraud_Probability'] < 0.3)]

# Calculate aggregate financial damage avoided (True Positive Volumes)
prevented_loss_val = true_positives['Amount_KES'].sum()

# Model sensitivity / capture rate
intercept_rate_pct = (len(true_positives) / (len(true_positives) + len(false_negatives))) * 100 if (len(true_positives) + len(false_negatives)) > 0 else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Total Ledger Evaluated</div>
        <div class='metric-value'>{total_inspected:,}</div>
        <div class='metric-subtext'>Continuous system ingestion</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Prevented Financial Damage</div>
        <div class='metric-value' style='color:#10b981;'>KES {prevented_loss_val/1e6:.2f}M</div>
        <div class='metric-subtext'>Volume of intercepted risk cases</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Fraud Interception Rate</div>
        <div class='metric-value' style='color:#38bdf8;'>{intercept_rate_pct:.2f}%</div>
        <div class='metric-subtext'>Recall rate against testing baseline</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    # False positive ratio (Alarms generated vs actual fraud cases identified)
    fp_ratio = len(false_positives) / len(true_positives) if len(true_positives) > 0 else 0
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Analyst Workload Ratio</div>
        <div class='metric-value'>{fp_ratio:.1f}:1</div>
        <div class='metric-subtext'>False alarms triggered per fraud case</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RISK ANALYTICS CHART BLOCK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_chart_left, col_chart_right = st.columns([1, 1], gap="large")

with col_chart_left:
    st.markdown("<h3 style='color:#f8fafc; font-size:1.0rem; text-transform:uppercase; letter-spacing:0.05em;'>Active Exposure by Attack Vector</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-top:0;'>Distribution of confirmed fraud patterns within the ledger.</p>", unsafe_allow_html=True)
    
    # Isolate true fraud cases to analyze the types
    fraud_distribution = df[df['Is_Fraud'] == 1].groupby('Fraud_Type')['Amount_KES'].agg(['count', 'sum']).reset_index()
    fraud_distribution.columns = ['Attack Vector', 'Transaction Count', 'Total Volume (KES)']
    
    fig_vector = px.bar(
        fraud_distribution,
        x='Total Volume (KES)',
        y='Attack Vector',
        orientation='h',
        color='Total Volume (KES)',
        color_continuous_scale='Reds',
        text_auto='.2s'
    )
    
    fig_vector.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        xaxis=dict(title="Volume (KES)", showgrid=True, gridcolor='#334155', tickfont=dict(color='#94a3b8')),
        yaxis=dict(title="", showgrid=False, tickfont=dict(color='#f8fafc', size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300
    )
    st.plotly_chart(fig_vector, use_container_width=True, config={'displayModeBar': False})

with col_chart_right:
    st.markdown("<h3 style='color:#f8fafc; font-size:1.0rem; text-transform:uppercase; letter-spacing:0.05em;'>Risk Density across Ingestion Channels</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-top:0;'>Comparative exposure of transfer paths by risk tiers.</p>", unsafe_allow_html=True)
    
    # Calculate volume of risk by channel
    channel_risk = df.groupby(['Channel', 'Risk_Tier'])['Amount_KES'].sum().reset_index()
    channel_risk.columns = ['Ingestion Channel', 'System Risk Tier', 'Total Volume (KES)']
    
    fig_channel = px.bar(
        channel_risk,
        x='Total Volume (KES)',
        y='Ingestion Channel',
        color='System Risk Tier',
        orientation='h',
        color_discrete_map={
            "High Risk (Critical)": "#ef4444",
            "Medium Risk (Suspicious)": "#facc15",
            "Low Risk (Verified)": "#10b981"
        },
        category_orders={"System Risk Tier": ["High Risk (Critical)", "Medium Risk (Suspicious)", "Low Risk (Verified)"]}
    )
    
    fig_channel.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Total Volume Evaluated (KES)", showgrid=True, gridcolor='#334155', tickfont=dict(color='#94a3b8')),
        yaxis=dict(title="", showgrid=False, tickfont=dict(color='#f8fafc', size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, font=dict(color='#94a3b8', size=10), title=""),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        barmode='stack'
    )
    st.plotly_chart(fig_channel, use_container_width=True, config={'displayModeBar': False})

# ─────────────────────────────────────────────────────────────────────────────
# LIVE FORENSIC INVESTIGATION LEDGER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#f8fafc; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.05em;'>Forensic Audit Ingestion Ledger</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:0.9rem; margin-top:0; margin-bottom:1rem;'>Interact with live transaction queues prioritized dynamically by the model's computed probability score.</p>", unsafe_allow_html=True)

# Select filters
col_filter1, col_filter2, col_filter3 = st.columns(3)
with col_filter1:
    selected_risk_tier = st.selectbox("Filter Risk Priority Tier:", ["All Inspected", "High Risk (Critical)", "Medium Risk (Suspicious)", "Low Risk (Verified)"])
with col_filter2:
    selected_channel = st.selectbox("Filter Ingestion Channel:", ["All Channels"] + list(df['Channel'].unique()))
with col_filter3:
    search_query = st.text_input("Search Customer ID or Transaction ID:")

# Filter Ingestion Dataset
filtered_df = df.copy()

if selected_risk_tier != "All Inspected":
    filtered_df = filtered_df[filtered_df['Risk_Tier'] == selected_risk_tier]
if selected_channel != "All Channels":
    filtered_df = filtered_df[filtered_df['Channel'] == selected_channel]
if search_query:
    filtered_df = filtered_df[
        filtered_df['Customer_ID'].str.contains(search_query, case=False, na=False) |
        filtered_df['Transaction_ID'].str.contains(search_query, case=False, na=False)
    ]

# Sort dynamically so most critical are strictly pushed to the top of the queue
filtered_df = filtered_df.sort_values(by='Fraud_Probability', ascending=False)

# Render Ingestion Ledger
ledger_display = filtered_df[['Transaction_ID', 'Customer_ID', 'Timestamp', 'Amount_KES', 'Location', 'Channel', 'Fraud_Probability', 'Risk_Tier']].head(20)

table_html = """
<table>
    <thead>
        <tr>
            <th>Transaction ID</th>
            <th>Customer ID</th>
            <th>Timestamp</th>
            <th>Amount (KES)</th>
            <th>Node Location</th>
            <th>Ingestion Channel</th>
            <th>Score</th>
            <th>Risk Allocation</th>
        </tr>
    </thead>
    <tbody>
"""

for _, row in ledger_display.iterrows():
    if row['Risk_Tier'] == "High Risk (Critical)":
        badge_style = "badge-critical"
    elif row['Risk_Tier'] == "Medium Risk (Suspicious)":
        badge_style = "badge-suspicious"
    else:
        badge_style = "badge-verified"
        
    table_html += (
        "<tr>"
        f"<td style='font-family:\"JetBrains Mono\", monospace; color:#38bdf8;'>{row['Transaction_ID']}</td>"
        f"<td style='font-family:\"JetBrains Mono\", monospace;'>{row['Customer_ID']}</td>"
        f"<td>{row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</td>"
        f"<td style='font-family:\"JetBrains Mono\", monospace; font-weight:600;'>{row['Amount_KES']:,.2f}</td>"
        f"<td>{row['Location']}</td>"
        f"<td>{row['Channel']}</td>"
        f"<td style='font-family:\"JetBrains Mono\", monospace;'>{row['Fraud_Probability']:.4f}</td>"
        f"<td><span class='decision-badge {badge_style}' style='padding:0.25rem 0.5rem; font-size:0.7rem;'>{row['Risk_Tier'].replace(' (Critical)','').replace(' (Suspicious)','').replace(' (Verified)','')}</span></td>"
        "</tr>"
    )

table_html += "</tbody></table>"

st.markdown(f"<div style='background:#1e293b; border:1px solid #334155; border-radius:6px; padding:1.25rem; overflow-x:auto;'>{table_html}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INDIVIDUAL FORENSIC FOCUS CASE INTERVIEW
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#f8fafc; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.05em;'>Forensic Deep-Dive Investigation</h3>", unsafe_allow_html=True)

audit_candidates = filtered_df[filtered_df['Fraud_Probability'] >= 0.3]['Transaction_ID'].tolist()

if audit_candidates:
    selected_audit_tx = st.selectbox("Select Flagged Transaction to Deep Audit:", audit_candidates)
    
    if selected_audit_tx:
        audit_data = df[df['Transaction_ID'] == selected_audit_tx].iloc[0]
        
        # Display forensic case metrics
        col_forensic_score, col_forensic_facts = st.columns([1, 2], gap="large")
        
        with col_forensic_score:
            prob_pct = audit_data['Fraud_Probability'] * 100
            if audit_data['Risk_Tier'] == "High Risk (Critical)":
                color_audit = "#ef4444"
                badge_class = "badge-critical"
                system_verdict = "DECISION: AUTOMATED HOLD PLACED"
            elif audit_data['Risk_Tier'] == "Medium Risk (Suspicious)":
                color_audit = "#facc15"
                badge_class = "badge-suspicious"
                system_verdict = "DECISION: ESCALATED FOR MANUAL REVIEW"
            else:
                color_audit = "#10b981"
                badge_class = "badge-verified"
                system_verdict = "DECISION: CLEARED FOR EXECUTION"
                
            st.markdown(f"""
            <div style='background:#1e293b; border:1px solid #334155; border-radius:6px; padding:2rem; text-align:center;'>
                <div style='color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1rem;'>Model Probability Score</div>
                <div style='color:{color_audit}; font-size:4rem; font-weight:700; font-family:"JetBrains Mono", monospace; line-height:1;'>{prob_pct:.1f}%</div>
                <div style='color:#f8fafc; font-size:1.0rem; font-weight:600; margin-top:1rem;'>{audit_data['Risk_Tier']}</div>
                <div style='color:#94a3b8; font-size:0.75rem; margin-top:0.5rem;'>Calibrated risk classification threshold: 30.0%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div class='decision-badge {badge_class}' style='margin-top:1rem; font-size:0.85rem;'>{system_verdict}</div>", unsafe_allow_html=True)

        with col_forensic_facts:
            st.markdown("<h4 style='color:#f8fafc; font-size:0.9rem; text-transform:uppercase; margin-bottom:1rem;'>Transactional Forensic Fact Sheet</h4>", unsafe_allow_html=True)
            
            # Prepare contextual data
            metrics_forensic = [
                "Transaction Amount", 
                "Historical Customer Average", 
                "Behavioral Deviation Ratio", 
                "Ingestion Path Location", 
                "Channel Signature", 
                "Behavioral Pattern Flag"
            ]
            
            vals_forensic = [
                f"KES {audit_data['Amount_KES']:,.2f}",
                f"KES {audit_data['Historical_Average_KES']:,.2f}",
                f"{audit_data['Amount_to_Historical_Ratio']:.2f}x standard volume",
                f"{audit_data['Location']} (Node Classification)",
                f"{audit_data['Channel']}",
                f"{audit_data['Fraud_Type'] if audit_data['Is_Fraud'] == 1 else 'None'}"
            ]
            
            audit_fact_df = pd.DataFrame({
                "Parameter": metrics_forensic,
                "Case Fact": vals_forensic
            })
            st.table(audit_fact_df)
else:
    st.info("System confirmation: No active alert cases currently present in the filtered ledger view.")
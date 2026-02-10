"""
SYNAPSE - Professional Risk Analysis System
Legacy Transaction Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Import custom modules
from modules.data_preprocess import load_and_clean_data, get_data_statistics
from modules.anomaly_detect import detect_anomalies, calculate_entity_risk
from modules.graph_builder import (
    build_transaction_graph, 
    detect_circular_flows, 
    identify_shell_entities,
    get_high_risk_subgraph,
    calculate_graph_metrics
)
from modules.report_generator import generate_anomaly_report, generate_summary_stats

# Page configuration
st.set_page_config(
    page_title="SYNAPSE | Analysis Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional aesthetic
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --accent-blue: #00f3ff;
        --dark-bg: #0d1117;
        --card-bg: #161b22;
        --border-color: #30363d;
    }
    
    /* Global styles */
    .stApp {
        background: var(--dark-bg);
    }
    
    /* Headers */
    h1 {
        color: #c9d1d9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    h2, h3 {
        color: #c9d1d9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Cards/containers */
    .stMetric {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 15px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# SYNAPSE")
st.markdown("### Transaction Risk Analysis Dashboard")
st.markdown("---")

# Ethical disclaimer
st.markdown("""
<div style="background: rgba(139, 148, 158, 0.1); border: 1px solid #30363d; border-radius: 6px; padding: 15px; text-align: center; color: #8b949e;">
    <strong>OPERATIONAL DISCLAIMER</strong><br>
    This system is designed for internal research and analytical evaluation purposes.<br>
    All data processing follows established anonymization protocols.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Select Section:",
        ["📊 Upload Data", "🔎 Analyze Transactions", "🕸️ Network Graph", "📥 Download Report"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **SYNAPSE** provides analytical depth for:
    - 💸 Anomaly detection
    - 🔄 Circular flow identification
    - 🏢 Relationship mapping
    - 📈 Composite risk scoring
    """)
    
    st.markdown("---")
    st.markdown("**Dhairya Singh Dhaila**")
    st.markdown("*Lead Analyst & Developer*")

# Initialize session state (reused from original)
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'report' not in st.session_state:
    st.session_state.report = None

# Page: Upload Data
if page == "📊 Upload Data":
    st.markdown("## 📊 Data Operations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Transaction Data Import")
        uploaded_file = st.file_uploader("Select CSV dataset", type=['csv'])
        
        if uploaded_file is not None:
            try:
                temp_path = "temp_upload.csv"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing records..."):
                    df = load_and_clean_data(temp_path)
                    st.session_state.data = df
                    os.remove(temp_path)
                
                st.success(f"Loaded {len(df)} transactions.")
                st.dataframe(df.head(10), use_container_width=True)
                
                stats = get_data_statistics(df)
                st.markdown("### 📈 Summary Statistics")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Transactions", f"{stats['total_transactions']:,}")
                col_b.metric("Countries", f"{stats['countries']:,}")
                col_c.metric("Avg Transaction", f"${stats['avg_transaction']:,.2f}")
                
            except Exception as e:
                st.error(f"Error loading records: {str(e)}")
    
    with col2:
        st.markdown("### 📝 Sample Data")
        if st.button("Load Standard Sample"):
            try:
                sample_path = "sample_data/transactions.csv"
                df = load_and_clean_data(sample_path)
                st.session_state.data = df
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Page: Analyze Transactions
elif page == "🔎 Analyze Transactions":
    st.markdown("## 🔎 Anomaly Detection")
    
    if st.session_state.data is None:
        st.warning("Please upload data to begin.")
    else:
        df = st.session_state.data
        contamination = st.slider("Anomaly sensitivity (%)", 1, 30, 10) / 100
        
        if st.button("Execute Analysis", type="primary"):
            with st.spinner("Processing..."):
                analyzed_df = detect_anomalies(df, contamination=contamination)
                st.session_state.analyzed_data = analyzed_df
                entity_risks = calculate_entity_risk(analyzed_df)
                G = build_transaction_graph(analyzed_df)
                st.session_state.graph = G
                cycles = detect_circular_flows(G)
                shell_entities = identify_shell_entities(G)
                st.session_state.report = generate_anomaly_report(
                    analyzed_df, entity_risks, cycles, shell_entities, output_path='anomalies.json'
                )
            st.success("Analysis complete.")
        
        if st.session_state.analyzed_data is not None:
            analyzed_df = st.session_state.analyzed_data
            st.markdown("---")
            st.markdown("### 📊 Results Overview")
            suspicious_count = analyzed_df['is_suspicious'].sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Analyzed", f"{len(analyzed_df):,}")
            col2.metric("Flagged Indicators", f"{suspicious_count:,}")
            col3.metric("Max Risk Score", f"{analyzed_df['risk_score'].max():.1f}")
            
            st.dataframe(
                analyzed_df[analyzed_df['is_suspicious']].nlargest(10, 'risk_score'),
                use_container_width=True
            )

# Page: Network Graph
elif page == "🕸️ Network Graph":
    st.markdown("## 🕸️ Network Visualization")
    if st.session_state.graph is None:
        st.warning("Please run analysis first.")
    else:
        G = st.session_state.graph
        st.markdown("### 📊 Network Profile")
        col1, col2 = st.columns(2)
        col1.metric("Network Nodes", G.number_of_nodes())
        col2.metric("Network Edges", G.number_of_edges())
        st.info("Interactive graph visualization would be rendered here.")

# Page: Download Report
elif page == "📥 Download Report":
    st.markdown("## 📥 Export Data")
    if st.session_state.report is None:
        st.warning("Please run analysis first.")
    else:
        st.success("Analytical report ready for export.")
        st.download_button(
            label="Download JSON Summary",
            data=json.dumps(st.session_state.report, indent=2),
            file_name="synapse_summary.json"
        )

# Footer
st.markdown("""
<div style='text-align: center; color: #8b949e; font-family: sans-serif; font-size: 0.8em;'>
    <p>📡 SYNAPSE Intelligence Framework | Dhairya Singh Dhaila</p>
    <p>© 2026 Analytical Research Systems</p>
</div>
""", unsafe_allow_html=True)

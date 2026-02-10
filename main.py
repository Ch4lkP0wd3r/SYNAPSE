"""
SYNAPSE - Intelligence Analysis System
Professional Transaction Monitoring & Risk Assessment Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# Import custom modules
from modules.data_preprocess import (
    load_and_clean_data, 
    get_data_statistics,
    generate_realistic_sample_data
)
from modules.anomaly_detect import detect_anomalies, calculate_entity_risk
from modules.graph_builder import (
    build_transaction_graph, 
    detect_circular_flows, 
    identify_shell_entities,
    get_high_risk_subgraph,
    calculate_graph_metrics
)
from modules.report_generator import generate_anomaly_report, generate_summary_stats

# PROFESSIONAL ANALYSIS MODULES
from modules.advanced_ml import (
    ensemble_anomaly_detection,
    calculate_transaction_velocity,
    detect_structuring,
    calculate_model_metrics
)
from modules.prediction_engine import (
    train_ultra_advanced_model,
    predict_with_ultra_model,
    print_model_performance
)
from modules.geographic_intel import (
    analyze_geographic_risk,
    detect_geographic_patterns,
    generate_geographic_heatmap_data,
    get_jurisdiction_risk_score
)
from modules.advanced_graph import (
    calculate_advanced_centrality,
    detect_communities,
    identify_key_players,
    detect_layering_schemes,
    find_suspicious_subgraphs,
    detect_fan_patterns
)

# Page configuration
st.set_page_config(
    page_title="SYNAPSE | Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Monochrome Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-color: #0d1117;
        --card-bg: #161b22;
        --border-color: #30363d;
        --text-color: #c9d1d9;
        --secondary-text: #8b949e;
        --accent: #f0f6fc;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
    
    .badge {
        background: var(--border-color);
        color: var(--text-color);
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 4px;
        display: inline-block;
        font-size: 0.8em;
        margin: 10px 0;
        border: 1px solid var(--border-color);
    }
    
    .stMetric {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 12px;
    }
    
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 4px;
        padding: 8px 16px;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }

    .stAlert {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>📡 SYNAPSE</h1>", unsafe_allow_html=True)
st.markdown("<div class='badge'>Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<h3>Professional Analysis & Risk Assessment Framework</h3>", unsafe_allow_html=True)
st.markdown("---")

# Ethical disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ ETHICAL DISCLAIMER ⚠️<br>
    This tool is for research and educational purposes only.<br>
    Not intended for real-world surveillance or enforcement.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "Select Section:",
        [
            "📊 Data Operations",
            "🔎 Multi-Layer Analysis",
            "🎯 High-Precision ML",
            "🌍 Geographic Intelligence",
            "🕸️ Network Intelligence",
            "📥 Summary Report"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**Lead Developer:**")
    st.markdown("Dhairya Singh Dhaila")
    st.markdown("*Specialist in Forensic Intelligence*")
    
    st.markdown("---")
    st.markdown("### ℹ️ Intelligence Policy")
    st.markdown("""
    **Core Capabilities:**
    - Composite Model Detection
    - Risk Scoring Engine
    - Forensic Network Analysis
    - Jurisdiction Assessment
    """)
    
    st.markdown("---")
    st.markdown("**SYNAPSE**")
    st.markdown("*Professional Analysis Framework*")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'report' not in st.session_state:
    st.session_state.report = None
if 'geo_analysis' not in st.session_state:
    st.session_state.geo_analysis = None
if 'centrality' not in st.session_state:
    st.session_state.centrality = None
if 'communities' not in st.session_state:
    st.session_state.communities = None
if 'ultra_model' not in st.session_state:
    st.session_state.ultra_model = None
if 'ultra_metrics' not in st.session_state:
    st.session_state.ultra_metrics = None

# Page: Data Operations
if page == "📊 Data Operations":
    st.markdown("## 📊 Data Operations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Transaction Data Import")
        st.markdown("Standardized format: `sender`, `receiver`, `amount`, `date`, `country`")
        
        uploaded_file = st.file_uploader(
            "Select dataset (CSV)",
            type=['csv'],
            help="Upload transaction data for analysis"
        )
        
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
                
                st.markdown("### 👁️ Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                stats = get_data_statistics(df)
                st.markdown("### 📈 Key Statistics")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Transactions", f"{stats['total_transactions']:,}")
                    st.metric("Unique Senders", f"{stats['unique_senders']:,}")
                with col_b:
                    st.metric("Unique Receivers", f"{stats['unique_receivers']:,}")
                    st.metric("Countries", f"{stats['countries']:,}")
                with col_c:
                    st.metric("Total Volume", f"${stats['total_volume']:,.2f}")
                    st.metric("Avg Transaction", f"${stats['avg_transaction']:,.2f}")
                
            except Exception as e:
                st.error(f"Error loading records: {str(e)}")
    
    with col2:
        st.markdown("### 📝 Sample Dataset")
        st.markdown("Initialize with standard sample data.")
        
        if st.button("Load Realistic Sample"):
            try:
                # Generate realistic data instead of reading static file
                df_raw = generate_realistic_sample_data(n_records=300)
                # Apply standard hashes and validation
                df = load_and_clean_data(df_raw) if isinstance(df_raw, str) else df_raw
                # We need to ensure sender_id/receiver_id exist since load_and_clean_data handles it for files
                if 'sender_id' not in df.columns:
                     df['sender_id'] = df['sender'].apply(lambda x: hash(str(x)) % 10000)
                     df['receiver_id'] = df['receiver'].apply(lambda x: hash(str(x)) % 10000)
                     df['transaction_id'] = range(len(df))
                
                st.session_state.data = df
                st.success(f"Realistic sample data ({len(df)} records) initialized.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Page: Multi-Layer Analysis
elif page == "🔎 Multi-Layer Analysis":
    st.markdown("## 🔎 Multi-Layer Risk Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data to begin analysis.")
    else:
        df = st.session_state.data
        
        st.markdown("### ⚙️ Advanced Analysis Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            contamination = st.slider(
                "Expected anomaly rate (%)",
                min_value=1,
                max_value=30,
                value=10,
                help="Percentage of transactions expected to be anomalous"
            ) / 100
            
        with col2:
            use_ensemble = st.checkbox(
                "Use Multi-Layer Ensemble", 
                value=True,
                help="Combines Isolation Forest, DBSCAN, and Statistical Z-Score"
            )
        
        if st.button("Run Multi-Layer Analysis", type="primary"):
            with st.spinner("Analyzing high-dimensional risk patterns..."):
                # Run ensemble detection
                if use_ensemble:
                    analyzed_df = ensemble_anomaly_detection(df, contamination=contamination)
                else:
                    analyzed_df = detect_anomalies(df, contamination=contamination)
                
                st.session_state.analyzed_data = analyzed_df
                
                # Calculate entity risks
                entity_risks = calculate_entity_risk(analyzed_df)
                
                # Build graph
                G = build_transaction_graph(analyzed_df)
                st.session_state.graph = G
                
                # Advanced analytics
                cycles = detect_circular_flows(G)
                shell_entities = identify_shell_entities(G)
                structuring_cases = detect_structuring(analyzed_df)
                velocity_metrics = calculate_transaction_velocity(analyzed_df)
                
                # Gen 4: Fan Patterns
                fan_patterns = detect_fan_patterns(G)
                
                # Geographic analysis
                geo_analysis = analyze_geographic_risk(analyzed_df)
                st.session_state.geo_analysis = geo_analysis
                
                # Generate report
                report = generate_anomaly_report(
                    analyzed_df,
                    entity_risks,
                    cycles,
                    shell_entities,
                    output_path='multi_layer_report.json'
                )
                
                # Add analysis metadata to report
                report['structuring_cases'] = structuring_cases
                report['velocity_metrics'] = velocity_metrics
                report['geographic_analysis'] = geo_analysis
                report['fan_patterns'] = fan_patterns
                
                st.session_state.report = report
                
            st.success("Analysis Complete.")
        
        # Show results
        if st.session_state.analyzed_data is not None:
            analyzed_df = st.session_state.analyzed_data
            
            st.markdown("---")
            st.markdown("### 📊 Multi-Layer Analysis Results")
            
            # Enhanced metrics
            suspicious_count = analyzed_df['is_suspicious'].sum()
            total_count = len(analyzed_df)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Analyzed", f"{total_count:,}")
            with col2:
                st.metric("Flagged", f"{suspicious_count:,}", 
                         delta=f"{(suspicious_count/total_count*100):.1f}%")
            with col3:
                if 'confidence_score' in analyzed_df.columns:
                    avg_conf = analyzed_df[analyzed_df['is_suspicious']]['confidence_score'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                else:
                    st.metric("Avg Risk", f"{analyzed_df['risk_score'].mean():.1f}")
            with col4:
                if 'threat_level' in analyzed_df.columns:
                    critical = (analyzed_df['threat_level'] == 'CRITICAL').sum()
                    st.metric("Critical Threats", f"{critical:,}")
                else:
                    st.metric("Max Risk", f"{analyzed_df['risk_score'].max():.1f}")
            with col5:
                if st.session_state.geo_analysis:
                    high_risk_geo = st.session_state.geo_analysis.get('high_risk_transactions', 0)
                    st.metric("High-Risk Jurisdictions", f"{high_risk_geo:,}")
                else:
                    st.metric("Countries", f"{analyzed_df['country'].nunique():,}")
            
            # Threat level breakdown
            if 'threat_level' in analyzed_df.columns:
                st.markdown("### 🚨 Threat Level Distribution")
                threat_counts = analyzed_df['threat_level'].value_counts()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    critical_count = threat_counts.get('CRITICAL', 0)
                    st.markdown(f"<div class='threat-critical'>🔴 CRITICAL: {critical_count}</div>", unsafe_allow_html=True)
                with col2:
                    high_count = threat_counts.get('HIGH', 0)
                    st.markdown(f"<div class='threat-high'>🟠 HIGH: {high_count}</div>", unsafe_allow_html=True)
                with col3:
                    medium_count = threat_counts.get('MEDIUM', 0)
                    st.info(f"🟡 MEDIUM: {medium_count}")
                with col4:
                    low_count = threat_counts.get('LOW', 0)
                    st.success(f"🟢 LOW: {low_count}")
            
            # Top threats
            st.markdown("### 🎯 Top Priority Targets")
            suspicious_df = analyzed_df[analyzed_df['is_suspicious']].nlargest(10, 'risk_score')
            
            display_cols = ['transaction_id', 'sender_id', 'receiver_id', 'amount', 'country', 'risk_score']
            if 'confidence_score' in analyzed_df.columns:
                display_cols.append('confidence_score')
            if 'threat_level' in analyzed_df.columns:
                display_cols.append('threat_level')
            
            st.dataframe(
                suspicious_df[display_cols].style.background_gradient(
                    subset=['risk_score'],
                    cmap='Reds'
                ),
                use_container_width=True
            )
            
            # Structuring detection results
            if st.session_state.report and 'structuring_cases' in st.session_state.report:
                structuring = st.session_state.report['structuring_cases']
                if structuring:
                    st.markdown("### 💰 Structuring Detection (Smurfing)")
                    st.warning(f"⚠️ Detected {len(structuring)} potential structuring schemes")
                    
                    for i, case in enumerate(structuring[:5]):
                        with st.expander(f"Case #{i+1}: Entity {case['sender_id']} - ${case['total_amount']:,.2f}"):
                            st.write(f"**Transactions:** {case['num_transactions']}")
                            st.write(f"**Time Window:** {case['time_window_hours']} hours")
                            st.write(f"**Average Amount:** ${case['avg_transaction']:,.2f}")
                            st.write(f"**Suspicion Score:** {case['suspicion_score']:.1f}/100")
            
            # Gen 4: Mathematical Anomalies
            st.markdown("### 🔢 Gen 4: Mathematical & Temporal Anomalies")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📉 Benford's Law Deviation")
                benford_flags = analyzed_df[analyzed_df['benford_deviation'] > 0.8].nlargest(5, 'benford_deviation')
                if not benford_flags.empty:
                    st.dataframe(benford_flags[['transaction_id', 'amount', 'benford_deviation']], use_container_width=True)
                else:
                    st.success("✅ No significant Benford's Law deviations found.")
            
            with col2:
                st.markdown("#### 😴 Dormant Account Awakenings")
                dormancy_flags = analyzed_df[analyzed_df['is_dormant_awakening'] == 1].nlargest(5, 'amount')
                if not dormancy_flags.empty:
                    st.dataframe(dormancy_flags[['transaction_id', 'sender_id', 'time_since_last']], use_container_width=True)
                else:
                    st.success("✅ No suspicious dormancy awakenings detected.")


# Page: High-Precision ML
elif page == "🎯 High-Precision ML":
    st.markdown("## 🎯 High-Precision ML Benchmarking")
    st.markdown("<div class='badge'>Statistical Ensemble Optimization</div>", unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.data
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🧠 Model Training")
            st.info("Train a supervised ensemble using XGBoost, LightGBM, and Random Forest.")
            
            test_split = st.slider("Test Set Split (%)", 10, 40, 20) / 100
            
            if st.button("Execute Training Cycle", type="primary"):
                with st.spinner("Optimizing ensemble parameters..."):
                    model_data = train_ultra_advanced_model(df, test_size=test_split)
                    st.session_state.ultra_model = model_data
                    st.session_state.ultra_metrics = model_data['metrics']
                st.success("Training complete.")
        
        with col2:
            st.markdown("### 📈 Model Performance")
            if st.session_state.ultra_metrics:
                metrics = st.session_state.ultra_metrics['ensemble']
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                    st.metric("Precision", f"{metrics['precision']*100:.2f}%")
                with m2:
                    st.metric("Recall", f"{metrics['recall']*100:.2f}%")
                    st.metric("F1 Score", f"{metrics['f1']*100:.2f}%")
            else:
                st.write("Train the model to see performance metrics.")
        
        if st.session_state.ultra_model:
            st.markdown("---")
            st.markdown("### 🔎 Deploy Ultra-Model for Detection")
            
            if st.button("🚀 Run Ultra-Detection"):
                with st.spinner("⚡ Processing transactions with 99% accuracy ensemble..."):
                    ultra_results = predict_with_ultra_model(df, st.session_state.ultra_model)
                    # Map fraud_probability to risk_score for UI compatibility
                    if 'fraud_probability' in ultra_results.columns:
                        ultra_results['risk_score'] = ultra_results['fraud_probability'] * 100
                    st.session_state.analyzed_data = ultra_results
                st.success("✅ Ultra-Detection Complete!")
                st.rerun()
            
            if st.session_state.analyzed_data is not None and 'fraud_probability' in st.session_state.analyzed_data.columns:
                st.markdown("### 🔴 Ultra-ML Flagged Transactions")
                flagged = st.session_state.analyzed_data[st.session_state.analyzed_data['is_suspicious']].nlargest(10, 'fraud_probability')
                
                st.dataframe(
                    flagged[['transaction_id', 'sender_id', 'receiver_id', 'amount', 'fraud_probability', 'confidence_score', 'threat_level']].style.background_gradient(
                        subset=['fraud_probability'],
                        cmap='RdYlGn_r',
                        vmin=0, vmax=1.0
                    ).format({'fraud_probability': '{:.4f}', 'confidence_score': '{:.1f}'}),
                    use_container_width=True
                )
                
                st.markdown("---")
                st.markdown("### 🧬 Composite Judgment (10 Categories)")
                st.info("Select a transaction to view forensic categorization markers.")
                
                selected_tid = st.selectbox("Select Transaction ID for Forensic Audit:", flagged['transaction_id'].tolist())
                
                if selected_tid:
                    case_data = flagged[flagged['transaction_id'] == selected_tid].iloc[0]
                    
                    # Prepare Radar Chart Data
                    categories = [
                        'Behavioral', 'Geographic', 'Temporal', 'Relationship', 
                        'Metadata', 'Consistency', 'Graph', 'Intelligence', 
                        'ML Indicators', 'Human Context'
                    ]
                    
                    values = [
                        case_data.get('composite_behavioral_score', 0),
                        case_data.get('composite_geographic_score', 0),
                        case_data.get('composite_temporal_score', 0),
                        case_data.get('composite_relationship_score', 0),
                        case_data.get('composite_metadata_score', 0),
                        case_data.get('composite_consistency_score', 0),
                        case_data.get('composite_graph_score', 0),
                        case_data.get('composite_intelligence_score', 0),
                        case_data.get('composite_ml_indicators_score', 0),
                        case_data.get('composite_human_context_score', 0)
                    ]
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Judgment Profile',
                        line=dict(color='#00f3ff', width=2),
                        fillcolor='rgba(0, 243, 255, 0.4)', # 40% transparent cyan
                        marker=dict(color='#00f3ff', size=8)
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, 
                                range=[0, 100],
                                gridcolor='rgba(255, 255, 255, 0.1)',
                                linecolor='rgba(255, 255, 255, 0.1)',
                                tickfont=dict(size=8, color='#888')
                            ),
                            angularaxis=dict(
                                gridcolor='rgba(255, 255, 255, 0.1)',
                                linecolor='rgba(255, 255, 255, 0.1)',
                                tickfont=dict(size=10, color='#e6edf3')
                            ),
                            bgcolor='rgba(0,0,0,0)' # Transparent background
                        ),
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#e6edf3',
                        height=450,
                        margin=dict(l=80, r=80, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Detailed Forensic Analysis
                    st.markdown("---")
                    col_narr, col_tactic = st.columns([1, 1])
                    
                    with col_narr:
                        st.markdown("#### 🔍 Objective Forensic Analysis")
                        narratives = case_data.get('composite_narrative_strings', "")
                        if narratives:
                            for r in narratives.split(" | "):
                                st.markdown(f"- ⚠️ **{r}**")
                        else:
                            st.success("Analysis complete: No anomalous signatures identified.")
                    
                    with col_tactic:
                        st.markdown("#### 🎭 Probable Tactic Interpretation")
                        tactics = case_data.get('composite_tactic_strings', "")
                        if tactics:
                            for t in tactics.split(" | "):
                                st.markdown(f"<div style='background-color:#2d333b; color:#e6edf3; padding:10px; border-radius:4px; margin-bottom:5px; border:1px solid #444c56;'>🛡️ {t}</div>", unsafe_allow_html=True)
                        else:
                            st.info("Routine operational pattern identified.")
                            
                    st.markdown("---")
                    st.markdown("### 🛠️ Strategic Recommendation")
                    if case_data['fraud_probability'] > 0.8:
                        st.error("🚨 **CRITICAL RISK INDICATORS**: Pattern consistency suggests advanced laundering involvement.")
                    elif case_data['fraud_probability'] > 0.5:
                        st.warning("⚠️ **EVIDENCE GATHERING**: Indicators present. Reviewing documentation is recommended.")
                    else:
                        st.success("✅ **STABLE**: Risk markers within acceptable bounds.")



# Page: Geographic Intelligence
elif page == "🌍 Geographic Intelligence":
    st.markdown("## 🌍 Geographic Intelligence Analysis")
    
    if st.session_state.geo_analysis is None:
        st.warning("Please run Multi-Layer Analysis first.")
    else:
        geo = st.session_state.geo_analysis
        
        st.markdown("### 🗺️ Geographic Risk Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Countries", f"{geo.get('total_countries', 0):,}")
        with col2:
            st.metric("High-Risk Txns", f"{geo.get('high_risk_transactions', 0):,}")
        with col3:
            st.metric("Tax Haven Txns", f"{geo.get('tax_haven_transactions', 0):,}")
        with col4:
            cross_border = geo.get('cross_border_rate', 0)
            st.metric("Cross-Border Rate", f"{cross_border*100:.1f}%")
        
        # Geographic patterns
        if st.session_state.data is not None:
            patterns = detect_geographic_patterns(st.session_state.data)
            
            if patterns:
                st.markdown("### 🚩 Detected Geographic Patterns")
                
                for pattern in patterns[:10]:
                    pattern_type = pattern['type']
                    
                    if pattern_type == 'HIGH_RISK_JURISDICTION':
                        st.error(f"🔴 **{pattern['country']}** ({pattern['risk_level']}): "
                                f"{pattern['num_transactions']} txns, ${pattern['total_amount']:,.2f}")
                    
                    elif pattern_type == 'TAX_HAVEN_ROUTING':
                        st.warning(f"🏴‍☠️ **Tax Haven: {pattern['jurisdiction']}**: "
                                  f"{pattern['num_transactions']} txns, ${pattern['total_amount']:,.2f}")
                    
                    elif pattern_type == 'GEOGRAPHIC_DISPERSION':
                        st.info(f"🌐 **Entity {pattern['entity_id']}** active in {pattern['num_countries']} countries")
        
        # Country heat map data
        st.markdown("### 🗺️ Geographic Heat Map")
        heatmap_data = generate_geographic_heatmap_data(st.session_state.data)
        
        if heatmap_data:
            # Create DataFrame for visualization
            heat_df = pd.DataFrame.from_dict(heatmap_data, orient='index').reset_index()
            heat_df.columns = ['country', 'risk_score', 'transaction_count', 'total_amount', 'avg_amount']
            heat_df = heat_df.nlargest(20, 'risk_score')
            
            # Bar chart
            fig = px.bar(
                heat_df,
                x='country',
                y='risk_score',
                color='risk_score',
                color_continuous_scale='Reds',
                title='Top 20 Countries by Risk Score',
                labels={'risk_score': 'Risk Score', 'country': 'Country'}
            )
            fig.update_layout(
                plot_bgcolor='#0a0e27',
                paper_bgcolor='#0a0e27',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

# Page: Network Intelligence
elif page == "🕸️ Network Intelligence":
    st.markdown("## 🕸️ Advanced Network Intelligence")
    
    if st.session_state.graph is None:
        st.warning("Please run Multi-Layer Analysis first.")
    else:
        G = st.session_state.graph
        
        # Calculate advanced metrics if not already done
        if st.session_state.centrality is None:
            with st.spinner("🔍 Calculating advanced network metrics..."):
                centrality = calculate_advanced_centrality(G)
                communities = detect_communities(G)
                st.session_state.centrality = centrality
                st.session_state.communities = communities
        
        centrality = st.session_state.centrality
        communities = st.session_state.communities
        
        # Network metrics
        st.markdown("### 📊 Network Intelligence Metrics")
        metrics = calculate_graph_metrics(G)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", f"{metrics['total_nodes']:,}")
        with col2:
            st.metric("Total Edges", f"{metrics['total_edges']:,}")
        with col3:
            st.metric("Communities", f"{communities.get('num_communities', 0):,}")
        with col4:
            st.metric("Network Density", f"{metrics['density']:.4f}")
        
        # Key players
        st.markdown("### 🎯 Key Players (High-Value Targets)")
        key_players = identify_key_players(G, centrality)
        
        if key_players:
            players_df = pd.DataFrame(key_players[:15])
            st.dataframe(
                players_df[['entity_id', 'role', 'influence_score', 'risk_score', 'connections']].style.background_gradient(
                    subset=['influence_score'],
                    cmap='Blues'
                ),
                use_container_width=True
            )
        
        # Community detection
        if communities and communities['num_communities'] > 0:
            st.markdown("### 🔗 Detected Criminal Networks")
            st.info(f"Identified {communities['num_communities']} distinct networks/communities")
            
            for i, comm in enumerate(communities['community_stats'][:5]):
                with st.expander(f"Network #{i+1}: {comm['size']} entities, ${comm['total_flow']:,.2f} total flow"):
                    st.write(f"**Density:** {comm['density']:.3f}")
                    st.write(f"**Avg Degree:** {comm['avg_degree']:.2f}")
                    st.write(f"**Key Members:** {', '.join(map(str, comm['nodes']))}")
        
        # Gen 4: Fan Patterns
        if st.session_state.report and 'fan_patterns' in st.session_state.report:
            fans = st.session_state.report['fan_patterns']
            if fans:
                st.markdown("### 🌪️ Gen 4: Fan-In & Fan-Out Patterns")
                for i, fan in enumerate(fans[:5]):
                    threat_color = "red" if fan['suspicion_score'] > 70 else "orange"
                    st.markdown(f"""
                    <div style='border-left: 5px solid {threat_color}; padding-left: 10px; margin-bottom: 5px;'>
                        <strong>{fan['type']}</strong>: Entity {fan['entity_id']} <br>
                        Connections: {fan['degree']} | Total Volume: ${fan['total_volume']:,.2f}
                    </div>
                    """, unsafe_allow_html=True)

        
        # Layering schemes
        layering = detect_layering_schemes(G, max_depth=4)
        if layering:
            st.markdown("### 🔄 Detected Layering Schemes")
            st.warning(f"⚠️ Found {len(layering)} complex layering patterns")
            
            for i, scheme in enumerate(layering[:5]):
                with st.expander(f"Scheme #{i+1}: {scheme['length']} hops, ${scheme['total_flow']:,.2f}"):
                    st.write(f"**Path:** {' → '.join(map(str, scheme['path']))}")
                    st.write(f"**Suspicion Score:** {scheme['suspicion_score']:.1f}/100")
        
        # Network visualization
        st.markdown("### 🕸️ Network Visualization")
        show_all = st.checkbox("Show all nodes", value=False)
        risk_threshold = st.slider("Risk threshold", 0, 100, 70)
        
        viz_graph = G if show_all else get_high_risk_subgraph(G, risk_threshold=risk_threshold)
        
        if viz_graph.number_of_nodes() > 0:
            pos = nx.spring_layout(viz_graph, k=0.5, iterations=50)
            
            edge_trace = go.Scatter(
                x=[], y=[],
                line=dict(width=0.5, color='rgba(0, 243, 255, 0.3)'),
                hoverinfo='none',
                mode='lines'
            )
            
            for edge in viz_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
            
            node_x, node_y, node_color, node_text = [], [], [], []
            
            for node in viz_graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                risk = viz_graph.nodes[node]['risk_score']
                node_color.append(risk)
                
                # Enhanced tooltip with centrality info
                influence = centrality.get(node, {}).get('influence_score', 0)
                role = 'UNKNOWN'
                for player in key_players:
                    if player['entity_id'] == node:
                        role = player['role']
                        break
                
                node_text.append(
                    f"Entity: {node}<br>"
                    f"Risk: {risk:.1f}<br>"
                    f"Role: {role}<br>"
                    f"Influence: {influence:.1f}<br>"
                    f"Sent: ${viz_graph.nodes[node]['total_sent']:,.0f}<br>"
                    f"Received: ${viz_graph.nodes[node]['total_received']:,.0f}"
                )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='Reds',
                    size=10,
                    color=node_color,
                    colorbar=dict(
                        thickness=15,
                        title=dict(text='Risk Score', side='right'),
                        xanchor='left'
                    ),
                    line=dict(width=2, color='white')
                )
            )
            
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(text='Criminal Network Map', font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='#0a0e27',
                    paper_bgcolor='#0a0e27',
                    font=dict(color='white')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Page: Intelligence Report
elif page == "📥 Intelligence Report":
    st.markdown("## 📥 Intelligence Summary Report")
    
    if st.session_state.report is None:
        st.warning("Please run Multi-Layer Analysis first.")
    else:
        report = st.session_state.report
        
        st.markdown("### 📊 Executive Summary")
        summary_stats = generate_summary_stats(report)
        
        col1, col2 = st.columns(2)
        with col1:
            for key, value in list(summary_stats.items())[:4]:
                st.metric(key, value)
        with col2:
            for key, value in list(summary_stats.items())[4:]:
                st.metric(key, value)
        
        st.markdown("---")
        
        # Download options
        st.markdown("### 💾 Export Intelligence Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            json_str = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="📄 Download Intelligence Report (JSON)",
                data=json_str,
                file_name=f"synapse_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            if st.session_state.analyzed_data is not None:
                suspicious_df = st.session_state.analyzed_data[
                    st.session_state.analyzed_data['is_suspicious']
                ]
                csv = suspicious_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Flagged Transactions CSV",
                    data=csv,
                    file_name=f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        
        # Report preview
        st.markdown("### 👁️ Intelligence Report Preview")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Top Threats", "Geographic Intel", "Network Intel"])
        
        with tab1:
            st.json(report['summary'])
        
        with tab2:
            st.markdown("#### Top 10 Priority Targets")
            if report['suspicious_transactions']:
                for i, txn in enumerate(report['suspicious_transactions'][:10]):
                    threat_emoji = "🔴" if txn['risk_score'] > 80 else "🟠" if txn['risk_score'] > 60 else "🟡"
                    with st.expander(f"{threat_emoji} Transaction {txn['transaction_id']} - Risk: {txn['risk_score']:.1f}"):
                        st.write(f"**Sender:** {txn['sender_id']}")
                        st.write(f"**Receiver:** {txn['receiver_id']}")
                        st.write(f"**Amount:** ${txn['amount']:,.2f}")
                        st.write(f"**Country:** {txn['country']}")
                        st.write(f"**Reasons:**")
                        for reason in txn['reasons']:
                            st.write(f"  - {reason}")
        
        with tab3:
            if 'geographic_analysis' in report:
                st.json(report['geographic_analysis'])
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Circular Flows")
                st.json(report['circular_flows'][:3])
            with col2:
                st.markdown("#### Shell Entities")
                st.json(report['potential_shell_entities'][:3])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #00f3ff; font-family: monospace;'>
    <p>📡 SYNAPSE Intelligence Framework | Dhairya Singh Dhaila</p>
    <p style='font-size: 0.8em; color: #888;'>
        Advanced Multi-Model Ensemble | Geographic Intelligence | Network Analytics
    </p>
</div>
""", unsafe_allow_html=True)

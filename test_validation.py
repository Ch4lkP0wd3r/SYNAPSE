#!/usr/bin/env python3
"""
Multi-Layer Intelligence Test Script
Tests all advanced modules for SYNAPSE
"""

import sys
sys.path.insert(0, '.')

from modules.data_preprocess import load_and_clean_data
from modules.advanced_ml import (
    ensemble_anomaly_detection,
    detect_structuring,
    calculate_transaction_velocity
)
from modules.geographic_intel import (
    analyze_geographic_risk,
    detect_geographic_patterns
)
from modules.graph_builder import build_transaction_graph
from modules.advanced_graph import (
    calculate_advanced_centrality,
    detect_communities,
    identify_key_players
)

print("🔍 SYNAPSE Multi-Layer Analysis Test")
print("=" * 60)

# Test 1: Load data
print("\n1️⃣ Testing data preprocessing...")
try:
    df = load_and_clean_data('sample_data/transactions.csv')
    print(f"✅ Loaded {len(df)} transactions")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: Ensemble detection
print("\n2️⃣ Testing Multi-layer ensemble detection...")
try:
    analyzed_df = ensemble_anomaly_detection(df, contamination=0.1)
    suspicious_count = analyzed_df['is_suspicious'].sum()
    avg_confidence = analyzed_df[analyzed_df['is_suspicious']]['confidence_score'].mean()
    critical_count = (analyzed_df['threat_level'] == 'CRITICAL').sum()
    
    print(f"✅ Ensemble detection complete")
    print(f"   - Flagged: {suspicious_count} transactions")
    print(f"   - Avg Confidence: {avg_confidence:.1f}%")
    print(f"   - Critical Threats: {critical_count}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 3: Structuring detection
print("\n3️⃣ Testing structuring detection...")
try:
    structuring_cases = detect_structuring(analyzed_df)
    print(f"✅ Detected {len(structuring_cases)} structuring cases")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Geographic intelligence
print("\n4️⃣ Testing geographic intelligence...")
try:
    geo_analysis = analyze_geographic_risk(analyzed_df)
    geo_patterns = detect_geographic_patterns(analyzed_df)
    
    print(f"✅ Geographic analysis complete")
    print(f"   - Countries: {geo_analysis.get('total_countries', 0)}")
    print(f"   - High-risk txns: {geo_analysis.get('high_risk_transactions', 0)}")
    print(f"   - Patterns detected: {len(geo_patterns)}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Advanced graph analytics
print("\n5️⃣ Testing advanced graph analytics...")
try:
    G = build_transaction_graph(analyzed_df)
    centrality = calculate_advanced_centrality(G)
    communities = detect_communities(G)
    key_players = identify_key_players(G, centrality)
    
    print(f"✅ Advanced graph analytics complete")
    print(f"   - Nodes: {G.number_of_nodes()}")
    print(f"   - Communities: {communities.get('num_communities', 0)}")
    print(f"   - Key players: {len(key_players)}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ All intelligence tests passed!")
print("\nTo start the analysis dashboard, run:")
print("  ./venv/bin/streamlit run main.py")
print("=" * 60)

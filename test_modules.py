#!/usr/bin/env python3
"""
Quick test script to verify SYNAPSE modules work correctly
"""

import sys
sys.path.insert(0, '.')

from modules.data_preprocess import load_and_clean_data, get_data_statistics
from modules.anomaly_detect import detect_anomalies, calculate_entity_risk
from modules.graph_builder import build_transaction_graph, detect_circular_flows, identify_shell_entities
from modules.report_generator import generate_anomaly_report

print("🔍 SYNAPSE Module Test")
print("=" * 50)

# Test 1: Load data
print("\n1️⃣ Testing data preprocessing...")
try:
    df = load_and_clean_data('sample_data/transactions.csv')
    print(f"✅ Loaded {len(df)} transactions")
    stats = get_data_statistics(df)
    print(f"   - Unique senders: {stats['unique_senders']}")
    print(f"   - Unique receivers: {stats['unique_receivers']}")
    print(f"   - Total volume: ${stats['total_volume']:,.2f}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: Anomaly detection
print("\n2️⃣ Testing anomaly detection...")
try:
    analyzed_df = detect_anomalies(df, contamination=0.1)
    suspicious_count = analyzed_df['is_suspicious'].sum()
    print(f"✅ Detected {suspicious_count} suspicious transactions")
    print(f"   - Suspicion rate: {(suspicious_count/len(df)*100):.1f}%")
    print(f"   - Max risk score: {analyzed_df['risk_score'].max():.1f}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 3: Graph building
print("\n3️⃣ Testing graph construction...")
try:
    G = build_transaction_graph(analyzed_df)
    print(f"✅ Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 4: Pattern detection
print("\n4️⃣ Testing pattern detection...")
try:
    cycles = detect_circular_flows(G)
    shell_entities = identify_shell_entities(G)
    print(f"✅ Detected {len(cycles)} circular flows")
    print(f"✅ Identified {len(shell_entities)} potential shell entities")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 5: Report generation
print("\n5️⃣ Testing report generation...")
try:
    entity_risks = calculate_entity_risk(analyzed_df)
    report = generate_anomaly_report(
        analyzed_df,
        entity_risks,
        cycles,
        shell_entities,
        output_path='test_report.json'
    )
    print(f"✅ Generated report with {len(report['suspicious_transactions'])} anomalies")
    print(f"   - High-risk entities: {report['summary']['high_risk_entities']}")
    print(f"   - Circular flows: {report['summary']['circular_flows_detected']}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ All tests passed! SYNAPSE is ready to run.")
print("\nTo start the dashboard, run:")
print("  ./venv/bin/streamlit run main.py")
print("=" * 50)

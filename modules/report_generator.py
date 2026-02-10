"""
Report Generator Module for SYNAPSE
Creates JSON reports of detected anomalies and suspicious patterns
"""

import json
from datetime import datetime
import pandas as pd

def generate_anomaly_report(df, entity_risks, cycles, shell_entities, output_path='anomalies.json'):
    """
    Generate comprehensive JSON report of all detected anomalies
    
    Args:
        df: Transaction DataFrame with anomaly flags
        entity_risks: Dictionary of entity risk scores
        cycles: List of detected circular flows
        shell_entities: List of potential shell entities
        output_path: Path to save JSON report
        
    Returns:
        dict: Report dictionary
    """
    
    # Filter suspicious transactions
    suspicious_txns = df[df['is_suspicious'] == True].copy()
    
    # Build transaction anomalies list
    transaction_anomalies = []
    for idx, row in suspicious_txns.iterrows():
        from modules.anomaly_detect import get_anomaly_reasons
        
        anomaly = {
            'transaction_id': int(row['transaction_id']),
            'sender_id': int(row['sender_id']),
            'receiver_id': int(row['receiver_id']),
            'amount': float(row['amount']),
            'date': str(row['date']),
            'country': str(row['country']),
            'risk_score': float(row['risk_score']),
            'reasons': get_anomaly_reasons(row)
        }
        transaction_anomalies.append(anomaly)
    
    # Sort by risk score
    transaction_anomalies.sort(key=lambda x: x['risk_score'], reverse=True)
    
    # Get top risky entities
    top_risky_entities = sorted(
        entity_risks.values(),
        key=lambda x: x['avg_risk_score'],
        reverse=True
    )[:20]
    
    # Convert to serializable format
    top_risky_entities = [
        {
            'entity_id': int(e['entity_id']),
            'type': e['type'],
            'total_transactions': int(e['total_transactions']),
            'suspicious_transactions': int(e['suspicious_transactions']),
            'suspicion_rate': float(e['suspicion_rate']),
            'avg_risk_score': float(e['avg_risk_score'])
        }
        for e in top_risky_entities
    ]
    
    # Format circular flows
    circular_flows = [
        {
            'nodes': [int(n) for n in cycle['nodes']],
            'length': cycle['length'],
            'total_flow': float(cycle['total_flow']),
            'avg_flow': float(cycle['avg_flow'])
        }
        for cycle in cycles
    ]
    
    # Format shell entities
    shell_entities_formatted = [
        {
            'entity_id': int(e['entity_id']),
            'total_received': float(e['total_received']),
            'total_sent': float(e['total_sent']),
            'pass_through_ratio': float(e['pass_through_ratio']),
            'in_degree': int(e['in_degree']),
            'out_degree': int(e['out_degree']),
            'risk_score': float(e['risk_score'])
        }
        for e in shell_entities
    ]
    
    # Build complete report
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'tool': 'SYNAPSE - Professional Risk Analysis System',
            'version': '3.1.0',
            'disclaimer': 'This report is for research and analytical evaluation purposes only. All data remains anonymous.'
        },
        'summary': {
            'total_transactions_analyzed': len(df),
            'suspicious_transactions': len(suspicious_txns),
            'suspicion_rate': len(suspicious_txns) / len(df) if len(df) > 0 else 0,
            'total_entities': len(entity_risks),
            'high_risk_entities': len([e for e in entity_risks.values() if e['avg_risk_score'] > 70]),
            'circular_flows_detected': len(cycles),
            'potential_shell_entities': len(shell_entities)
        },
        'suspicious_transactions': transaction_anomalies[:100],  # Top 100
        'high_risk_entities': top_risky_entities,
        'circular_flows': circular_flows,
        'potential_shell_entities': shell_entities_formatted[:20]  # Top 20
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def generate_summary_stats(report):
    """
    Generate human-readable summary statistics
    
    Args:
        report: Report dictionary
        
    Returns:
        dict: Summary statistics
    """
    summary = report['summary']
    
    stats = {
        'Total Transactions': f"{summary['total_transactions_analyzed']:,}",
        'Suspicious Transactions': f"{summary['suspicious_transactions']:,}",
        'Suspicion Rate': f"{summary['suspicion_rate']*100:.2f}%",
        'Total Entities': f"{summary['total_entities']:,}",
        'High-Risk Entities': f"{summary['high_risk_entities']:,}",
        'Circular Flows': f"{summary['circular_flows_detected']:,}",
        'Shell Entities': f"{summary['potential_shell_entities']:,}"
    }
    
    return stats

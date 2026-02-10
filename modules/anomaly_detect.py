"""
Anomaly Detection Module for SYNAPSE
Uses Isolation Forest to detect suspicious transaction patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(df, contamination=0.1):
    """
    Detect anomalous transactions using Isolation Forest
    
    Args:
        df: Transaction DataFrame
        contamination: Expected proportion of anomalies (default 0.1 = 10%)
        
    Returns:
        pd.DataFrame: DataFrame with anomaly scores and flags
    """
    
    # Feature engineering for anomaly detection
    features_df = df.copy()
    
    # Calculate transaction frequency per sender
    sender_freq = df.groupby('sender_id').size()
    features_df['sender_frequency'] = features_df['sender_id'].map(sender_freq)
    
    # Calculate transaction frequency per receiver
    receiver_freq = df.groupby('receiver_id').size()
    features_df['receiver_frequency'] = features_df['receiver_id'].map(receiver_freq)
    
    # Calculate average amount per sender
    sender_avg = df.groupby('sender_id')['amount'].mean()
    features_df['sender_avg_amount'] = features_df['sender_id'].map(sender_avg)
    
    # Calculate average amount per receiver
    receiver_avg = df.groupby('receiver_id')['amount'].mean()
    features_df['receiver_avg_amount'] = features_df['receiver_id'].map(receiver_avg)
    
    # Amount deviation from sender's average
    features_df['amount_deviation'] = abs(features_df['amount'] - features_df['sender_avg_amount'])
    
    # Round-number detection (common in money laundering)
    features_df['is_round_number'] = features_df['amount'].apply(
        lambda x: 1 if x % 1000 == 0 or x % 500 == 0 else 0
    )
    
    # Benford's Law Analysis
    def calculate_benford_deviation(amount):
        if amount <= 0: return 0
        first_digit = int(str(amount).replace('0.', '').replace('.', '')[0])
        benford_expected = {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}
        return 1.0 - benford_expected.get(first_digit, 0.1)
    
    features_df['benford_deviation'] = features_df['amount'].apply(calculate_benford_deviation)
    
    # Dormancy & Time Analysis
    if 'date' in features_df.columns:
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df = features_df.sort_values(['sender_id', 'date'])
        features_df['time_since_last'] = features_df.groupby('sender_id')['date'].diff().dt.total_seconds() / (24 * 3600)
        features_df['is_dormant_awakening'] = (features_df['time_since_last'] > 30).astype(int)
    else:
        features_df['benford_deviation'] = 0
        features_df['time_since_last'] = 0
        features_df['is_dormant_awakening'] = 0
    
    # Select features for Isolation Forest
    feature_columns = [
        'amount',
        'sender_frequency',
        'receiver_frequency',
        'sender_avg_amount',
        'receiver_avg_amount',
        'amount_deviation',
        'is_round_number'
    ]
    
    X = features_df[feature_columns].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    # Predict anomalies (-1 = anomaly, 1 = normal)
    features_df['anomaly'] = iso_forest.fit_predict(X_scaled)
    
    # Get anomaly scores (lower = more anomalous)
    features_df['anomaly_score'] = iso_forest.score_samples(X_scaled)
    
    # Convert to risk score (0-100, higher = riskier)
    min_score = features_df['anomaly_score'].min()
    max_score = features_df['anomaly_score'].max()
    features_df['risk_score'] = 100 * (max_score - features_df['anomaly_score']) / (max_score - min_score + 1e-10)
    
    # Confidence score (0-100)
    features_df['confidence_score'] = features_df['risk_score'] * 0.8 # Baseline confidence
    
    # Flag as suspicious if anomaly detected
    features_df['is_suspicious'] = features_df['anomaly'] == -1
    
    # Threat level classification
    features_df['threat_level'] = pd.cut(
        features_df['risk_score'],
        bins=[0, 30, 60, 80, 100],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )
    
    return features_df

def get_anomaly_reasons(row):
    """
    Generate human-readable reasons for why a transaction is flagged
    
    Args:
        row: DataFrame row
        
    Returns:
        list: List of reason strings
    """
    reasons = []
    
    if row['risk_score'] > 70:
        reasons.append("High risk score")
    
    if row['amount'] > row['sender_avg_amount'] * 3:
        reasons.append("Amount significantly higher than sender's average")
    
    if row['sender_frequency'] > 50:
        reasons.append("High transaction frequency from sender")
    
    if row['receiver_frequency'] > 50:
        reasons.append("High transaction frequency to receiver")
    
    if row['is_round_number'] == 1 and row['amount'] >= 10000:
        reasons.append("Large round-number transaction (common in layering)")
    
    if row['amount_deviation'] > row['sender_avg_amount'] * 2:
        reasons.append("Unusual amount pattern for this sender")
    
    if not reasons:
        reasons.append("Anomalous pattern detected by ML model")
    
    return reasons

def calculate_entity_risk(df):
    """
    Calculate risk scores for each entity (sender/receiver)
    
    Args:
        df: Transaction DataFrame with anomaly flags
        
    Returns:
        dict: Entity risk scores
    """
    entity_risks = {}
    
    # Calculate sender risks
    for sender_id in df['sender_id'].unique():
        sender_txns = df[df['sender_id'] == sender_id]
        suspicious_count = sender_txns['is_suspicious'].sum()
        total_count = len(sender_txns)
        avg_risk = sender_txns['risk_score'].mean()
        
        entity_risks[f"sender_{sender_id}"] = {
            'entity_id': sender_id,
            'type': 'sender',
            'total_transactions': total_count,
            'suspicious_transactions': int(suspicious_count),
            'suspicion_rate': suspicious_count / total_count if total_count > 0 else 0,
            'avg_risk_score': avg_risk
        }
    
    # Calculate receiver risks
    for receiver_id in df['receiver_id'].unique():
        receiver_txns = df[df['receiver_id'] == receiver_id]
        suspicious_count = receiver_txns['is_suspicious'].sum()
        total_count = len(receiver_txns)
        avg_risk = receiver_txns['risk_score'].mean()
        
        entity_risks[f"receiver_{receiver_id}"] = {
            'entity_id': receiver_id,
            'type': 'receiver',
            'total_transactions': total_count,
            'suspicious_transactions': int(suspicious_count),
            'suspicion_rate': suspicious_count / total_count if total_count > 0 else 0,
            'avg_risk_score': avg_risk
        }
    
    return entity_risks

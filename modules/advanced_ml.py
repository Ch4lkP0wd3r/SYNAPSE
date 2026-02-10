"""
Advanced ML Module for SYNAPSE - Multi-Layer Detection
Multi-model ensemble with Random Forest, DBSCAN, and confidence scoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def ensemble_anomaly_detection(df, contamination=0.1):
    """
    Multi-layer ensemble detection using multiple ML models
    
    Args:
        df: Transaction DataFrame with features
        contamination: Expected anomaly rate
        
    Returns:
        pd.DataFrame: Enhanced detection with confidence scores
    """
    
    # Feature engineering (same as before)
    features_df = df.copy()
    
    sender_freq = df.groupby('sender_id').size()
    features_df['sender_frequency'] = features_df['sender_id'].map(sender_freq)
    
    receiver_freq = df.groupby('receiver_id').size()
    features_df['receiver_frequency'] = features_df['receiver_id'].map(receiver_freq)
    
    sender_avg = df.groupby('sender_id')['amount'].mean()
    features_df['sender_avg_amount'] = features_df['sender_id'].map(sender_avg)
    
    receiver_avg = df.groupby('receiver_id')['amount'].mean()
    features_df['receiver_avg_amount'] = features_df['receiver_id'].map(receiver_avg)
    
    features_df['amount_deviation'] = abs(features_df['amount'] - features_df['sender_avg_amount'])
    
    features_df['is_round_number'] = features_df['amount'].apply(
        lambda x: 1 if x % 1000 == 0 or x % 500 == 0 else 0
    )
    
    # Additional multi-layer features
    features_df['log_amount'] = np.log1p(features_df['amount'])
    features_df['amount_to_avg_ratio'] = features_df['amount'] / (features_df['sender_avg_amount'] + 1)
    
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
        
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['hour'] = features_df['date'].dt.hour if hasattr(features_df['date'].dt, 'hour') else 12
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    else:
        features_df['benford_deviation'] = 0
        features_df['time_since_last'] = 0
        features_df['is_dormant_awakening'] = 0
        features_df['day_of_week'] = 0
        features_df['hour'] = 12
        features_df['is_weekend'] = 0
    
    # Select features for models
    feature_columns = [
        'amount',
        'sender_frequency',
        'receiver_frequency',
        'sender_avg_amount',
        'receiver_avg_amount',
        'amount_deviation',
        'is_round_number',
        'log_amount',
        'amount_to_avg_ratio',
        'day_of_week',
        'is_weekend'
    ]
    
    X = features_df[feature_columns].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model 1: Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto'
    )
    iso_predictions = iso_forest.fit_predict(X_scaled)
    iso_scores = iso_forest.score_samples(X_scaled)
    
    # Model 2: DBSCAN Clustering (outliers are anomalies)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_predictions = np.where(dbscan_labels == -1, -1, 1)  # -1 = outlier/anomaly
    
    # Model 3: Statistical outlier detection (Z-score based)
    z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / (X_scaled.std(axis=0) + 1e-10))
    max_z_scores = z_scores.max(axis=1)
    z_threshold = np.percentile(max_z_scores, (1 - contamination) * 100)
    z_predictions = np.where(max_z_scores > z_threshold, -1, 1)
    
    # Ensemble voting (majority vote)
    votes = np.column_stack([iso_predictions, dbscan_predictions, z_predictions])
    ensemble_predictions = np.where(np.sum(votes == -1, axis=1) >= 2, -1, 1)
    
    # Confidence score (0-100): based on model agreement
    confidence = np.abs(np.sum(votes == -1, axis=1) - 1.5) / 1.5 * 100
    confidence = np.clip(confidence, 0, 100)
    
    # Risk score calculation (enhanced)
    min_score = iso_scores.min()
    max_score = iso_scores.max()
    risk_score = 100 * (max_score - iso_scores) / (max_score - min_score + 1e-10)
    
    # Boost risk score based on ensemble agreement
    risk_score = risk_score * (1 + confidence / 200)
    risk_score = np.clip(risk_score, 0, 100)
    
    # Add results to dataframe
    features_df['ensemble_prediction'] = ensemble_predictions
    features_df['is_suspicious'] = ensemble_predictions == -1
    features_df['risk_score'] = risk_score
    features_df['confidence_score'] = confidence
    features_df['iso_score'] = iso_scores
    features_df['dbscan_label'] = dbscan_labels
    features_df['z_score'] = max_z_scores
    
    # Threat level classification
    features_df['threat_level'] = pd.cut(
        features_df['risk_score'],
        bins=[0, 30, 60, 80, 100],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )
    
    return features_df

def calculate_transaction_velocity(df):
    """
    Calculate transaction velocity metrics (multi-layer temporal analysis)
    
    Args:
        df: Transaction DataFrame with date column
        
    Returns:
        dict: Velocity metrics
    """
    if 'date' not in df.columns:
        return {}
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate time differences between transactions
    df['time_diff'] = df.groupby('sender_id')['date'].diff().dt.total_seconds() / 3600  # hours
    
    velocity_metrics = {
        'avg_time_between_txns': df['time_diff'].mean(),
        'min_time_between_txns': df['time_diff'].min(),
        'rapid_fire_count': (df['time_diff'] < 1).sum(),  # < 1 hour apart
        'burst_transactions': (df['time_diff'] < 0.1).sum(),  # < 6 minutes apart
    }
    
    return velocity_metrics

def detect_structuring(df, threshold=10000, window=24):
    """
    Detect structuring (smurfing) - breaking large amounts into smaller transactions
    
    Args:
        df: Transaction DataFrame
        threshold: Reporting threshold (e.g., $10,000)
        window: Time window in hours
        
    Returns:
        list: Potential structuring cases
    """
    if 'date' not in df.columns:
        return []
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    structuring_cases = []
    
    # Group by sender
    for sender_id in df['sender_id'].unique():
        sender_txns = df[df['sender_id'] == sender_id].copy()
        
        if len(sender_txns) < 2:
            continue
        
        # Check for multiple transactions just below threshold
        for i in range(len(sender_txns)):
            window_start = sender_txns.iloc[i]['date']
            window_end = window_start + pd.Timedelta(hours=window)
            
            window_txns = sender_txns[
                (sender_txns['date'] >= window_start) & 
                (sender_txns['date'] <= window_end)
            ]
            
            if len(window_txns) >= 2:
                total_amount = window_txns['amount'].sum()
                below_threshold = (window_txns['amount'] < threshold).all()
                
                if total_amount >= threshold and below_threshold:
                    structuring_cases.append({
                        'sender_id': sender_id,
                        'num_transactions': len(window_txns),
                        'total_amount': total_amount,
                        'time_window_hours': window,
                        'avg_transaction': window_txns['amount'].mean(),
                        'start_date': window_start,
                        'end_date': window_end,
                        'suspicion_score': min(100, (total_amount / threshold) * 50)
                    })
    
    # Remove duplicates and sort by suspicion score
    unique_cases = []
    seen = set()
    for case in structuring_cases:
        key = (case['sender_id'], case['start_date'])
        if key not in seen:
            seen.add(key)
            unique_cases.append(case)
    
    unique_cases.sort(key=lambda x: x['suspicion_score'], reverse=True)
    
    return unique_cases[:20]  # Top 20 cases

def calculate_model_metrics(df):
    """
    Calculate performance metrics for the ensemble model
    
    Args:
        df: DataFrame with predictions
        
    Returns:
        dict: Model performance metrics
    """
    metrics = {
        'total_analyzed': len(df),
        'flagged_suspicious': df['is_suspicious'].sum(),
        'flagged_rate': df['is_suspicious'].mean(),
        'avg_confidence': df['confidence_score'].mean(),
        'high_confidence_flags': (df['confidence_score'] > 70).sum(),
        'critical_threats': (df['threat_level'] == 'CRITICAL').sum(),
        'high_threats': (df['threat_level'] == 'HIGH').sum(),
        'medium_threats': (df['threat_level'] == 'MEDIUM').sum(),
        'low_threats': (df['threat_level'] == 'LOW').sum(),
    }
    
    return metrics

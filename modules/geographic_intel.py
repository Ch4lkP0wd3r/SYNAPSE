"""
Geographic Intelligence Module for SYNAPSE - Multi-Layer Analysis
Cross-border tracking, high-risk jurisdiction detection, and geographic clustering
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# High-risk jurisdictions (based on FATF and international AML standards)
HIGH_RISK_JURISDICTIONS = {
    'CRITICAL': [
        'North Korea', 'Iran', 'Myanmar', 'Syria'
    ],
    'HIGH': [
        'Afghanistan', 'Albania', 'Barbados', 'Burkina Faso', 'Cambodia',
        'Cayman Islands', 'Haiti', 'Jamaica', 'Jordan', 'Mali', 'Morocco',
        'Nicaragua', 'Pakistan', 'Panama', 'Philippines', 'Senegal',
        'South Sudan', 'Uganda', 'Yemen', 'Zimbabwe'
    ],
    'ELEVATED': [
        'BVI', 'Seychelles', 'Cyprus', 'Malta', 'Luxembourg',
        'Monaco', 'Liechtenstein', 'Andorra', 'Bahamas', 'Bermuda'
    ]
}

# Tax havens and offshore financial centers
TAX_HAVENS = [
    'Cayman Islands', 'BVI', 'Bermuda', 'Luxembourg', 'Switzerland',
    'Singapore', 'Hong Kong', 'Monaco', 'Liechtenstein', 'Panama',
    'Bahamas', 'Malta', 'Cyprus', 'Seychelles', 'Mauritius'
]

def analyze_geographic_risk(df):
    """
    Analyze geographic risk patterns in transactions
    
    Args:
        df: Transaction DataFrame with country column
        
    Returns:
        dict: Geographic risk analysis
    """
    if 'country' not in df.columns:
        return {}
    
    df = df.copy()
    
    # Classify jurisdictions
    def classify_jurisdiction(country):
        for risk_level, countries in HIGH_RISK_JURISDICTIONS.items():
            if country in countries:
                return risk_level
        if country in TAX_HAVENS:
            return 'TAX_HAVEN'
        return 'STANDARD'
    
    df['jurisdiction_risk'] = df['country'].apply(classify_jurisdiction)
    
    # Calculate geographic metrics
    geo_analysis = {
        'total_countries': df['country'].nunique(),
        'high_risk_transactions': (df['jurisdiction_risk'].isin(['CRITICAL', 'HIGH'])).sum(),
        'tax_haven_transactions': (df['jurisdiction_risk'] == 'TAX_HAVEN').sum(),
        'cross_border_rate': calculate_cross_border_rate(df),
        'country_distribution': df['country'].value_counts().to_dict(),
        'risk_distribution': df['jurisdiction_risk'].value_counts().to_dict(),
        'high_risk_volume': df[df['jurisdiction_risk'].isin(['CRITICAL', 'HIGH'])]['amount'].sum(),
        'tax_haven_volume': df[df['jurisdiction_risk'] == 'TAX_HAVEN']['amount'].sum(),
    }
    
    return geo_analysis

def calculate_cross_border_rate(df):
    """
    Calculate rate of cross-border transactions
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        float: Cross-border transaction rate
    """
    if 'country' not in df.columns or len(df) == 0:
        return 0.0
    
    # Assume cross-border if sender and receiver countries differ
    # For this implementation, we'll use a heuristic
    unique_countries = df['country'].nunique()
    if unique_countries > 1:
        return min(1.0, unique_countries / len(df) * 2)
    return 0.0

def detect_geographic_patterns(df):
    """
    Detect suspicious geographic patterns
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        list: Detected geographic patterns
    """
    patterns = []
    
    if 'country' not in df.columns:
        return patterns
    
    df = df.copy()
    
    # Pattern 1: Rapid geographic dispersion (same entity, multiple countries)
    for entity_id in df['sender_id'].unique():
        entity_txns = df[df['sender_id'] == entity_id]
        countries = entity_txns['country'].unique()
        
        if len(countries) >= 5:
            patterns.append({
                'type': 'GEOGRAPHIC_DISPERSION',
                'entity_id': entity_id,
                'num_countries': len(countries),
                'countries': list(countries),
                'total_amount': entity_txns['amount'].sum(),
                'suspicion_score': min(100, len(countries) * 15)
            })
    
    # Pattern 2: Tax haven routing
    tax_haven_txns = df[df['country'].isin(TAX_HAVENS)]
    if len(tax_haven_txns) > 0:
        for haven in TAX_HAVENS:
            haven_txns = tax_haven_txns[tax_haven_txns['country'] == haven]
            if len(haven_txns) >= 3:
                patterns.append({
                    'type': 'TAX_HAVEN_ROUTING',
                    'jurisdiction': haven,
                    'num_transactions': len(haven_txns),
                    'total_amount': haven_txns['amount'].sum(),
                    'unique_entities': haven_txns['sender_id'].nunique(),
                    'suspicion_score': min(100, len(haven_txns) * 10)
                })
    
    # Pattern 3: High-risk jurisdiction activity
    for risk_level in ['CRITICAL', 'HIGH']:
        if risk_level in HIGH_RISK_JURISDICTIONS:
            for country in HIGH_RISK_JURISDICTIONS[risk_level]:
                country_txns = df[df['country'] == country]
                if len(country_txns) > 0:
                    patterns.append({
                        'type': 'HIGH_RISK_JURISDICTION',
                        'risk_level': risk_level,
                        'country': country,
                        'num_transactions': len(country_txns),
                        'total_amount': country_txns['amount'].sum(),
                        'suspicion_score': 90 if risk_level == 'CRITICAL' else 70
                    })
    
    # Sort by suspicion score
    patterns.sort(key=lambda x: x['suspicion_score'], reverse=True)
    
    return patterns[:15]  # Top 15 patterns

def generate_geographic_heatmap_data(df):
    """
    Generate data for geographic heat map visualization
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        dict: Heat map data by country
    """
    if 'country' not in df.columns:
        return {}
    
    # Aggregate by country
    country_data = df.groupby('country').agg({
        'amount': ['sum', 'count', 'mean'],
        'transaction_id': 'count'
    }).reset_index()
    
    country_data.columns = ['country', 'total_amount', 'transaction_count', 'avg_amount', 'txn_count']
    
    # Calculate risk scores
    heatmap_data = {}
    for _, row in country_data.iterrows():
        country = row['country']
        
        # Base risk on jurisdiction classification
        if country in HIGH_RISK_JURISDICTIONS.get('CRITICAL', []):
            base_risk = 90
        elif country in HIGH_RISK_JURISDICTIONS.get('HIGH', []):
            base_risk = 70
        elif country in TAX_HAVENS:
            base_risk = 60
        else:
            base_risk = 30
        
        # Adjust based on volume
        volume_factor = min(30, np.log1p(row['total_amount']) / 10)
        
        heatmap_data[country] = {
            'risk_score': min(100, base_risk + volume_factor),
            'transaction_count': int(row['transaction_count']),
            'total_amount': float(row['total_amount']),
            'avg_amount': float(row['avg_amount'])
        }
    
    return heatmap_data

def analyze_cross_border_flows(df):
    """
    Analyze cross-border transaction flows
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        list: Cross-border flow analysis
    """
    if 'country' not in df.columns:
        return []
    
    # Group by country pairs (assuming sender and receiver might be in different countries)
    # For this implementation, we'll analyze country-based flows
    
    flows = []
    country_groups = df.groupby('country')
    
    for country, group in country_groups:
        if len(group) > 1:
            flows.append({
                'country': country,
                'inflow_count': len(group),
                'total_volume': group['amount'].sum(),
                'avg_transaction': group['amount'].mean(),
                'unique_senders': group['sender_id'].nunique(),
                'unique_receivers': group['receiver_id'].nunique()
            })
    
    # Sort by volume
    flows.sort(key=lambda x: x['total_volume'], reverse=True)
    
    return flows

def get_jurisdiction_risk_score(country):
    """
    Get risk score for a specific jurisdiction
    
    Args:
        country: Country name
        
    Returns:
        int: Risk score (0-100)
    """
    if country in HIGH_RISK_JURISDICTIONS.get('CRITICAL', []):
        return 95
    elif country in HIGH_RISK_JURISDICTIONS.get('HIGH', []):
        return 75
    elif country in HIGH_RISK_JURISDICTIONS.get('ELEVATED', []):
        return 60
    elif country in TAX_HAVENS:
        return 55
    else:
        return 25

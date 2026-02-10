"""
Data Preprocessing Module for SYNAPSE
Handles CSV loading, cleaning, and validation of transaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(file_path):
    """
    Load transaction data from CSV and perform cleaning
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pd.DataFrame: Cleaned transaction data
    """
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['sender', 'receiver', 'amount', 'date', 'country']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['sender', 'receiver', 'amount'])
        
        # Clean amount column - ensure numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df[df['amount'] > 0]  # Remove negative or zero amounts
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Anonymize IDs - hash sender and receiver
        df['sender_id'] = df['sender'].apply(lambda x: hash(str(x)) % 10000)
        df['receiver_id'] = df['receiver'].apply(lambda x: hash(str(x)) % 10000)
        
        # Fill missing countries
        df['country'] = df['country'].fillna('UNKNOWN')
        
        # Add transaction ID
        df['transaction_id'] = range(len(df))
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def simulate_transaction_metadata(df):
    """
    Simulate advanced metadata for transactions (Composite Framework)
    to support comprehensive risk criteria.
    """
    np.random.seed(42)
    n = len(df)
    
    # 1. Device & Network Metadata
    df['device_id'] = [hashlib.md5(f"device_{np.random.randint(0, 1000)}".encode()).hexdigest()[:12] for _ in range(n)]
    df['ip_address'] = [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n)]
    df['vpn_detected'] = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    df['proxy_detected'] = np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    df['channel'] = np.random.choice(['MOBILE', 'WEB', 'API', 'ATM'], size=n, p=[0.5, 0.3, 0.1, 0.1])
    
    # 2. Entity Contextual Metadata
    sender_metadata = {}
    for sender in df['sender_id'].unique():
        sender_metadata[sender] = {
            'kyc_status': np.random.choice(['VERIFIED', 'PENDING', 'UNVERIFIED'], p=[0.8, 0.15, 0.05]),
            'occupation': np.random.choice(['ENGINEER', 'DOCTOR', 'SELF_EMPLOYED', 'UNEMPLOYED', 'STUDENT'], p=[0.3, 0.2, 0.3, 0.1, 0.1]),
            'income_estimate': np.random.randint(20000, 500000),
            'declared_business_purpose': np.random.choice(['PERSONAL', 'RETAIL', 'CONSULTING', 'CRYPTO', 'GAMES'], p=[0.4, 0.3, 0.1, 0.1, 0.1])
        }
    
    df['sender_kyc'] = df['sender_id'].map(lambda x: sender_metadata[x]['kyc_status'])
    df['sender_income'] = df['sender_id'].map(lambda x: sender_metadata[x]['income_estimate'])
    df['sender_purpose'] = df['sender_id'].map(lambda x: sender_metadata[x]['declared_business_purpose'])
    
    # 3. Behavioral Metadata
    df['auth_failures'] = np.random.choice([0, 1, 2, 3, 5], size=n, p=[0.9, 0.05, 0.02, 0.02, 0.01])
    df['session_duration'] = np.random.randint(10, 1200, size=n) # seconds
    
    return df

def validate_and_anonymize(df):
    """
    Validates, anonymizes, and enhances data with metadata simulation
    """
    # Assuming 'validate_data' and 'anonymize_entities' are conceptual steps
    # already handled by load_and_clean_data or are part of the input df
    # For this example, we'll assume the input df is already "validated" and "anonymized"
    # in terms of core columns and IDs, and we proceed to simulate metadata.
    
    # If actual validation/anonymization functions were intended, they would need to be defined.
    # For now, we directly apply metadata simulation.
    df = simulate_transaction_metadata(df)
    return df

def get_data_statistics(df):
    """
    Calculate basic statistics about the transaction data
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'total_transactions': len(df),
        'unique_senders': df['sender_id'].nunique(),
        'unique_receivers': df['receiver_id'].nunique(),
        'total_volume': df['amount'].sum(),
        'avg_transaction': df['amount'].mean(),
        'max_transaction': df['amount'].max(),
        'min_transaction': df['amount'].min(),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
        'countries': df['country'].nunique()
    }
    
    return stats
def generate_realistic_sample_data(n_records=200):
    """
    Generates a realistic synthetic transaction dataset with complex patterns.
    """
    np.random.seed(42)
    
    # 1. Base Data
    senders = [f"ENTITY_{np.random.randint(1000, 9999)}" for _ in range(50)]
    receivers = [f"ENTITY_{np.random.randint(1000, 9999)}" for _ in range(80)]
    countries = ['US', 'GB', 'DE', 'CA', 'FR', 'SG', 'JP', 'KY', 'KP', 'IR', 'VG', 'CH', 'LU', 'PA']
    
    data = []
    base_date = datetime(2026, 1, 1)
    
    for i in range(n_records):
        sender = np.random.choice(senders)
        receiver = np.random.choice(receivers)
        
        # More chaotic amount distribution - mixture of log-normals
        if np.random.random() > 0.8:
            amount = np.random.lognormal(mean=11, sigma=1.5) # Large corporate/money transfer
        else:
            amount = np.random.lognormal(mean=6, sigma=1.2) # Retail/Personal
            
        amount = np.round(amount, 2)
        
        # Jitter the date significantly
        days_offset = np.random.randint(0, 45)
        hours_offset = np.random.randint(0, 24)
        minutes_offset = np.random.randint(0, 60)
        timestamp = base_date + pd.to_timedelta(days_offset, unit='D') + \
                    pd.to_timedelta(hours_offset, unit='h') + \
                    pd.to_timedelta(minutes_offset, unit='m')
                    
        # Randomized country profile
        country_probs = [0.15, 0.1, 0.1, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.00] # Adjusted to match countries list length
        # Fix: ensure country_probs matches countries length
        country_probs = np.random.dirichlet(np.ones(len(countries)), size=1)[0]
        country = np.random.choice(countries, p=country_probs)
        
        data.append({
            'sender': sender,
            'receiver': receiver,
            'amount': amount,
            'date': timestamp,
            'country': country
        })
    
    # 2. Inject Specific Patterns with more noise
    
    # Pattern A: Enhanced Structuring (Smurfing)
    structurer = "ENTITY_SMURF_99"
    for i in range(np.random.randint(5, 12)):
        data.append({
            'sender': structurer,
            'receiver': "ENTITY_HUB_01",
            'amount': np.random.uniform(8500, 9950), 
            'date': base_date + pd.to_timedelta(15, unit='D') + pd.to_timedelta(i * 45, unit='m'),
            'country': 'US'
        })
        
    # Pattern B: High-Value Layering
    laundry_source = "ENTITY_WASH_44"
    for j in range(3):
        data.append({
            'sender': laundry_source,
            'receiver': f"SHELL_OFFSHORE_{j}",
            'amount': np.random.choice([250000, 500000, 750000]),
            'date': base_date + pd.to_timedelta(30, unit='D'),
            'country': np.random.choice(['KY', 'VG', 'PA'])
        })
        
    # Pattern C: Round Figures & Rapid Burst
    burst_sender = "ENTITY_BURST_77"
    for i in range(4):
        data.append({
            'sender': burst_sender,
            'receiver': "ENTITY_EXTERN_66",
            'amount': 25000, # Round
            'date': base_date + pd.to_timedelta(20, unit='D') + pd.to_timedelta(i*60, unit='s'),
            'country': 'VG' # British Virgin Islands
        })

    df = pd.DataFrame(data)
    
    # Correct date sorting
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

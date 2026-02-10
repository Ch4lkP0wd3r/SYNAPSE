#!/usr/bin/env python3
"""
High-Precision ML Benchmark Script
Validates the performance of the Supervised Ensemble
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from modules.data_preprocess import load_and_clean_data
from modules.prediction_engine import train_ultra_advanced_model, print_model_performance

print("🚀 SYNAPSE HIGH-PRECISION BENCHMARK")
print("=" * 60)

# 1. Load data
print("\n1️⃣ Loading transaction data...")
df = load_and_clean_data('sample_data/transactions.csv')
print(f"✅ Loaded {len(df)} transactions")

# 2. Train High-Precision Model
print("\n2️⃣ Training High-Precision Supervised Ensemble...")
print("   (XGBoost + LightGBM + Random Forest + Gradient Boosting)")
model_data = train_ultra_advanced_model(df, test_size=0.2)

# 3. Print Performance
print_model_performance(model_data['metrics'])

# 4. Accuracy Verification
ensemble_accuracy = model_data['metrics']['ensemble']['accuracy']
print(f"\nFinal Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")

if ensemble_accuracy >= 0.95:
    print("\n✅ HIGH-PRECISION TARGET MET! 🏆")
else:
    print(f"\n⚠️ Accuracy is {ensemble_accuracy*100:.2f}%. Optimization may be needed for specific datasets.")

print("=" * 60)

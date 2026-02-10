#!/bin/bash
# SYNAPSE Startup Script

echo "🔍 Starting SYNAPSE - Professional Risk Analysis System"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "📦 Installing dependencies..."
    ./venv/bin/pip install -q -r requirements.txt
fi

# Start Streamlit
echo "🚀 Launching dashboard..."
echo ""
./venv/bin/streamlit run main.py

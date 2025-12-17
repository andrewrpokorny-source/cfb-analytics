#!/bin/bash

PROJECT_DIR="/Users/andrewpokorny/cfb-analytics"
VENV_PATH="$PROJECT_DIR/venv/bin/activate"
DATE=$(date +"%Y-%m-%d")
OUTPUT_FILE="$PROJECT_DIR/reports/report_$DATE.txt"

mkdir -p "$PROJECT_DIR/reports"

echo "=========================================="
echo "🏈 CFB ALGO PIPELINE - STARTING ($DATE)"
echo "=========================================="

source "$VENV_PATH"
cd "$PROJECT_DIR"

echo "Step 1: Updating Data & Calculating Decay..."
python3 features.py

echo "Step 2: Retraining Leak-Proof Models..."
python3 model.py

echo "Step 3: Generating Live Predictions..."
# Capture output to file AND print to screen
python3 predict.py | tee "$OUTPUT_FILE"

echo ""
echo "✅ Pipeline Complete."
echo "Full report saved to: $OUTPUT_FILE"
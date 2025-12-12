#!/bin/bash

# --- CONFIGURATION ---
# Path to your project folder
PROJECT_DIR="/Users/andrewpokorny/cfb-analytics"
VENV_PATH="$PROJECT_DIR/venv/bin/activate"
DATE=$(date +"%Y-%m-%d")
OUTPUT_FILE="$PROJECT_DIR/reports/report_$DATE.txt"

# Create a reports folder if it doesn't exist
mkdir -p "$PROJECT_DIR/reports"

echo "=========================================="
echo "🏈 CFB ALGO PIPELINE - STARTING ($DATE)"
echo "=========================================="

# 1. Activate Virtual Environment
source "$VENV_PATH"
cd "$PROJECT_DIR"

# 2. Update Data & Features (Smart Decay)
echo ""
echo "Step 1: Updating Data & Calculating Decay..."
python3 features.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: features.py failed. Aborting."
    exit 1
fi

# 3. Retrain Models (The Brain)
echo ""
echo "Step 2: Retraining Leak-Proof Models..."
python3 model.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: model.py failed. Aborting."
    exit 1
fi

# 4. Generate Predictions
echo ""
echo "Step 3: Generating Live Predictions..."
python3 predict.py > "$OUTPUT_FILE"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: predict.py failed. Aborting."
    exit 1
fi

# 5. Display "Green Light" Bets (Confidence > 55%)
echo ""
echo "=========================================="
echo "💰 GREEN LIGHT BETS (>55% Confidence)"
echo "=========================================="
echo ""

# We use grep to filter the output file for lines containing high percentages
# (Looking for 55%, 56%, 57%, etc.)
grep -E "5[5-9]\.|6[0-9]\.|7[0-9]\." "$OUTPUT_FILE" | head -n 15

echo ""
echo "✅ Pipeline Complete."
echo "Full report saved to: $OUTPUT_FILE"
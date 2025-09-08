#!/bin/bash

# Run the full data processing and modelling pipeline

set -e

ROOT_DIR="$(dirname \"$0\")"

echo "[1/7] Cleaning raw data and converting timestamps..."
python "$ROOT_DIR/data_processor/preprocess.py"

echo "[2/7] Summarising news articles..."
python "$ROOT_DIR/data_processor/summarize.py"

echo "[3/7] Scoring news summaries with an LLM (ensure API key is set)..."
python "$ROOT_DIR/data_processor/score_by_gpt.py"

echo "[4/7] Integrating price and sentiment data..."
python "$ROOT_DIR/data_processor/price_news_integrate.py"

echo "[5/7] Training models..."
python "$ROOT_DIR/scripts/train_models.py" --config "$ROOT_DIR/configs/config_lstm.yaml"
python "$ROOT_DIR/scripts/train_models.py" --config "$ROOT_DIR/configs/config_gru.yaml"
python "$ROOT_DIR/scripts/train_models.py" --config "$ROOT_DIR/configs/config_transformer.yaml"

echo "[6/7] Evaluating models..."
python "$ROOT_DIR/scripts/evaluate_models.py"

echo "[7/7] Generating plots..."
python "$ROOT_DIR/scripts/generate_plots.py"

echo "Pipeline execution completed. Check the outputs/ directory for results."

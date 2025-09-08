# Stock Market Prediction Integrating LLM‑Generated Sentiment

This repository contains a reproducible template for the MSc Data Science extended research project described report.  It follows a modular layout and includes a dedicated **data_processor** folder inspired by the structure found in the referenced GitHub repository.  The goal of this project is to explore whether adding sentiment extracted from news using large language models (LLMs) can improve stock price prediction accuracy.

## Repository Structure

```text
stock_market_prediction/
├─ data_processor/                 # Raw and pre‑processed data plus scripts to prepare sentiment features
│  ├─ gpt_sentiment_price_news_integrate/
│  ├─ news_data_preprocessed/
│  ├─ news_data_raw/
│  ├─ news_data_sentiment_scored_by_gpt/
│  ├─ news_data_summarized/
│  ├─ stock_price_data_preprocessed/
│  ├─ stock_price_data_raw/
│  ├─ data_processor.md            # Documentation for the data processing pipeline
│  ├─ preprocess.py                # Convert raw CSV timestamps to UTC and normalise date formats
│  ├─ summarize.py                 # Summarise news text using LSA summarisation
│  ├─ score_by_gpt.py              # Use an LLM to assign sentiment scores to summarised news (requires API key)
│  └─ price_news_integrate.py      # Merge price and news data into a single DataFrame
├─ src/
│  ├─ data_processing.py           # Helper functions to prepare features for modelling
│  └─ models/
│     ├─ __init__.py
│     ├─ model_lstm.py             # PyTorch implementation of an LSTM network
│     ├─ model_gru.py              # PyTorch implementation of a GRU network
│     └─ model_transformer.py      # PyTorch implementation of a simple Transformer for time series
├─ scripts/
│  ├─ train_models.py              # Train all models defined in src/models using the merged dataset
│  ├─ evaluate_models.py           # Evaluate trained models on a held‑out set
│  └─ generate_plots.py            # Produce simple plots comparing predictions to actual values
├─ configs/
│  ├─ config_lstm.yaml             # Example hyper‑parameters for LSTM
│  ├─ config_gru.yaml              # Example hyper‑parameters for GRU
│  └─ config_transformer.yaml      # Example hyper‑parameters for Transformer
├─ outputs/
│  ├─ figures/                     # Generated plots will be saved here
│  └─ results/                     # Tables of metrics and predictions
├─ notebooks/                      # Optional: place exploratory notebooks here
├─ requirements.txt                # Python dependencies
└─ run_all.sh                      # Convenience script to run the full pipeline end‑to‑end
```

### data_processor

The `data_processor` folder mirrors the structure found in the referenced repository and contains scripts to prepare your dataset:

1. **`preprocess.py`** – reads raw CSV files from `news_data_raw/` and `stock_price_data_raw/`, converts date/time strings to UTC, sorts by date and writes the cleaned data to `news_data_preprocessed/` and `stock_price_data_preprocessed/`.  This step ensures consistent timestamps across all data sources.
2. **`summarize.py`** – uses an extractive summarisation algorithm (LSA via the [sumy](https://github.com/miso-belica/sumy) library) to produce concise summaries of each news article.  The summaries are stored in `news_data_summarized/` and are used as input to the sentiment scoring step.
3. **`score_by_gpt.py`** – calls a large language model via the OpenAI API to assign a sentiment score (1–5) to each summarised news snippet.  Because network access is disabled in this environment, this script contains a placeholder where you must supply your own API key and run externally.  The scored data are saved into `news_data_sentiment_scored_by_gpt/`.
4. **`price_news_integrate.py`** – merges the pre‑processed stock price data with the sentiment‑scored news data.  It aligns dates, fills missing values using a decay function and produces a merged dataset in `gpt_sentiment_price_news_integrate/`.  This merged file is the input to the modelling pipeline.

### src and models

The `src` directory contains helper functions and model implementations.  The three neural network classes defined in `src/models/` (LSTM, GRU and Transformer) follow standard PyTorch patterns.  You can adjust their architectures or add additional models if required.

### scripts

Scripts in the `scripts` folder orchestrate training and evaluation.  They read configuration files from the `configs` directory to set hyper‑parameters.  The training script saves trained model weights, the evaluation script computes metrics such as MAE and MSE, and the plotting script produces simple charts comparing predicted and actual prices.

### Running the pipeline

First install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run the data processing pipeline followed by model training and evaluation:

```bash
cd stock_market_prediction

# 1. Clean and normalise timestamps
python data_processor/preprocess.py

# 2. Summarise news articles
python data_processor/summarize.py

# 3. Score summarised news with an LLM – set your OpenAI API key in score_by_gpt.py before running
python data_processor/score_by_gpt.py

# 4. Integrate stock and news data
python data_processor/price_news_integrate.py

# 5. Train models
python scripts/train_models.py --config configs/config_lstm.yaml
python scripts/train_models.py --config configs/config_gru.yaml
python scripts/train_models.py --config configs/config_transformer.yaml

# 6. Evaluate models
python scripts/evaluate_models.py

# 7. Generate plots
python scripts/generate_plots.py
```

Alternatively, execute `./run_all.sh` from the repository root to run all steps in sequence.  See `data_processor/data_processor.md` for further details on each preprocessing stage.

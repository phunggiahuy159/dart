# DART: Dynamic Topic Modeling with Decompositional Forecasting and Context-Aware Global Alignment

Anonymous submission for dynamic topic modeling research.

## Overview

DART (Decompositional and Aligned Representation for Topics) is a novel neural dynamic topic model that addresses the critical limitation of topic identity drift in existing approaches. Our model enforces temporal coherence through structured forecasting and context-aware global alignment, enabling more reliable topic evolution tracking over time.

## Features

- **Decompositional Forecasting**: Models topic embedding evolution as time series with trend and seasonal decomposition
- **Global Beta Alignment (GBA)**: Maintains topic identity across time using context-aware TF-IDF masking
- **Temporal Coherence**: Significantly reduces topic identity drift while preserving natural semantic evolution
- **Comprehensive Evaluation**: Includes topic quality, temporal coherence, and downstream task performance metrics

## Requirements

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- PyTorch 2.6.0+ (with CUDA support recommended)
- gensim 4.3.3
- numpy 2.3.2
- pandas 2.3.1
- scikit-learn 1.7.1
- scipy 1.16.1
- tqdm 4.67.1



## Quick Start

Run the default experiment with NYT dataset:

```bash
python main.py
```

This will:
1. Download the NYT dataset automatically
2. Train the DART model with default parameters
3. Evaluate the model and display comprehensive metrics
4. Save top words for each time period to `top_words.txt`

## Dataset Configuration

### Available Datasets

The model supports multiple benchmark datasets used in the paper:

- **NYT**: New York Times articles (2012-2022) - Default dataset
- **NeurIPS**: NeurIPS conference publications (1987-2017)
- **ACL**: ACL Anthology articles (1973-2006)
- **UN**: United Nations session transcripts (1970-2015)
- **WHO**: WHO articles on non-pharmacological interventions (Jan-May 2020)

### Switching Datasets

To use a different dataset, modify the dataset name in `main.py`:

```python
# Change this line in main.py
download_dataset('DATASET_NAME', cache_path='./datasets')

# And update the dataset directory
dataset_dir = "./datasets/DATASET_NAME"
```

Replace `DATASET_NAME` with one of: `NYT`, `NeurIPS`, `ACL`, `UN`, `WHO`

### Custom Dataset

To use your own dataset, prepare it in the TopMost format and place it in the `./datasets/` directory. The dataset should contain:
- Document texts
- Timestamps for each document
- Vocabulary mappings
- Pre-processed features (TF-IDF, word embeddings)

## Training Configuration

### Basic Training

The model uses the following default hyperparameters optimized for best performance:

```python
model = DART(
    vocab_size=dataset.vocab_size,
    num_times=dataset.num_times,
    pretrained_WE=dataset.pretrained_WE,
    doc_tfidf=dataset.doc_tfidf,
    train_time_wordfreq=dataset.train_time_wordfreq,
    num_topics=50,                # Number of topics
    en_units=200,                 # Encoder hidden units
    weight_neg=7e+7,              # Negative sampling weight
    weight_pos=1.0,               # Positive alignment weight
    weight_beta_align=100,        # Global Beta Alignment weight
    beta_temp=1,                  # Temperature parameter
    dropout=0.01,                 # Dropout rate
)
```

### Training Parameters

```python
trainer = DynamicTrainer(
    model,
    dataset,
    epochs=300,           # Training epochs
    learning_rate=0.002,  # Learning rate
    batch_size=200,       # Batch size
    log_interval=5,       # Logging frequency
    verbose=True          # Verbose output
)
```

### Advanced Configuration

To modify hyperparameters, edit the model initialization in `main.py`:

- `num_topics`: Number of topics to discover (default: 50)
- `epochs`: Training epochs (default: 300)
- `learning_rate`: Optimizer learning rate (default: 0.002)
- `weight_beta_align`: Weight for Global Beta Alignment loss (default: 100)
- `weight_neg`: Weight for negative contrastive learning (default: 7e+7)

## Output Files

After training, the model generates:

- `top_words.txt`: Top words for each topic at each time period
- Console output with all evaluation metrics
- Trained model weights (can be saved by modifying the trainer)

## GPU Support

The model automatically uses CUDA if available. To force CPU usage, change:

```python
device = 'cpu'  # Instead of 'cuda'
```

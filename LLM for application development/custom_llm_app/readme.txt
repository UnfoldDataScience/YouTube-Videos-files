# Building a Custom LLM with Hugging Face

A comprehensive tutorial project demonstrating how to build, train, and deploy a custom Large Language Model using Hugging Face Transformers for customer support text classification.

## 🎯 Project Overview

This project shows you how to:
- Fine-tune a pre-trained model (DistilBERT) for text classification
- Create a robust training pipeline
- Evaluate model performance comprehensively  
- Deploy the model as a REST API
- Build a complete MLOps workflow

## 📁 Project Structure

```
custom-llm-app/
├── README.md
├── requirements.txt
├── data/
│   └── customer_support_sample.csv
├── scripts/
│   ├── train.py              # Complete training script
│   ├── evaluate.py           # Comprehensive evaluation
│   ├── simple_api.py         # FastAPI deployment
│   
├── models/                  # Saved models directory
├── outputs/                # Training outputs and logs
```
## 🚀 Quick Start

``### 1. Environment Setup

```bash

# Create virtual environment
python -m venv llm_env
source llm_env/bin/activate  # On Windows: llm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python scripts/train.py
```

This will:
- Create sample training data (or use your custom dataset)
- Fine-tune DistilBERT for customer support classification
- Save the trained model to `models/custom_support_model_working/`

### 3. Evaluate Performance

```bash
python scripts/evaluate.py
```

Generates:
- Detailed performance metrics
- Confidence score analysis


### 4. Deploy the API

```bash
python scripts/simple_api.py
```

The API will be available at `http://localhost:8000`

### 5. Test the API

```bash
# Single classification
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "I need help with billing"}'

# Batch classification
curl -X POST "http://localhost:8000/classify-batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Billing issue", "App crashed", "Great service!"]}'
```


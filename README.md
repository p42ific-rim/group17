# Knowledge Graph Link Prediction (Group 17)

This project is a comprehensive Knowledge Graph Embedding (KGE) framework designed for link prediction on the WN18RR dataset. It integrates structural graph learning with Natural Language Processing (NLP) to improve how machines understand and predict relationships between concepts.

---
## Theory

In a Knowledge Graph, facts are stored as triples: (Head Entity, Relation, Tail Entity). 
Our models map these into a low-dimensional vector space where the mathematical 
distance between vectors represents the likelihood of a relationship being 
true.

1. IMPLEMENTED MODELS:
We compare three distinct mathematical approaches to graph reasoning:
* TransE (Translational Distance): Views relations as translations in vector space (h + r ≈ t).
* DistMult (Semantic Matching): Uses bilinear diagonal interactions to measure the compatibility between entities.
* R-GCN (Relational Graph Convolutional Network): A Graph Neural Network (GNN) that uses Message Passing to let nodes learn from their neighbors.

---

## Key Features

HIGH-PERFORMANCE TRAINING:
* 1-N Training Strategy: The model scores a (Head, Relation) pair against all 40,000+ entities simultaneously using Binary Cross Entropy (BCE).
* Reciprocal Augmentation: Automatically generates inverse relations to allow the model to learn bidirectional reasoning.
* Vectorized Evaluation: Implements Filtered MRR (Mean Reciprocal Rank), ensuring true triples from the training set don't unfairly penalize the model during testing.

SEMANTIC ENRICHMENT (NLP INTEGRATION):
Unlike standard models that only look at "dots and lines," this project 
incorporates textual context:
* Definition Encoding: Uses Sentence-BERT (all-MiniLM-L6-v2) to convert WordNet text definitions into high-dimensional vectors.
* Automated Edge Discovery: Calculates cosine similarity between all entity definitions.
* Graph Augmentation: Automatically connects entities with highly similar definitions, providing "semantic shortcuts" for the GNN.

---
## Project structue

* run_experiment.py: The main orchestrator managing data loading, enrichment, and the model training loop.
* models/: Implementation of TransE, DistMult, and R-GCN architectures.
* trainer/trainer_bce.py: Optimized training engine with Label Smoothing and device-aware (MPS/CUDA/CPU) logic.
* trainer/evaluator.py: Industry-standard link prediction evaluation protocol.
* trainer/earlystopping.py: Monitor that saves the best model state to prevent overfitting.
* metrics/enrichment.py: NLP utilities for Sentence-BERT encoding and safe semantic similarity edge addition with Top-K capping.

---

## Setup

### 1. Prerequisites
* **Python:** 3.10 or 3.11 (Required for PyG compatibility)
* **Hardware:** NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU.

### 2. Environment Setup
Create and activate a virtual environment to isolate dependencies:

**macOS / Linux**
```bash
python3 -m venv kg-env
source kg-env/bin/activate
```

**Windows (PowerShell)**
```bash
python -m venv kg-env
.\kg-env\Scripts\activate
```

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

### 4. Install PyTorch (Hardware Accelerated)

**macOS / Linux**
OR for macOS (Apple Silicon M1/M2/M3)
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```

**Windows (PowerShell)**
For Windows and Linux (NVIDIA GPU support - CUDA 12.1)
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install PyG extensions (GPU-Compatible)

**macOS / Linux**
OR for macOS (Apple Silicon M1/M2/M3)
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
```

**Windows (PowerShell)**
For Windows and Linux (NVIDIA GPU support - CUDA 12.1)
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

### 6. Install torch-geometric and other dependencies
```bash
pip install torch-geometric numpy tqdm pandas
```

### 7. Install for enrichment run 

```bash
pip install sentence-transformers
```

```bash
pip install torch-geometric transformers scikit-learn
```

```bash
pip install --upgrade torch sentence-transformers
```

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html
```

```bash
pip install nltk
```

### (optional) 8. Verify Hardware Acceleration:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('MPS Available:', torch.backends.mps.is_available())"
```

---

## Usage

### Run Training & Evaluation and save traceback a log-file
```bash
./run_and_log.sh
```  

### Run Training & Evaluation without enrichment
```bash
python3 run_experiment.py
```     

### To switch between enrichment and no enrichment go to run_experiment.py and change the boolean in line: main(enrich=True)

---

## Evaluation Metrics

The project outputs standard KGE metrics:
* MRR (Mean Reciprocal Rank): Average of the inverse ranks of true entities.
* Hits@1, @3, and @10: Percentage of times the true entity appeared in the 
  top 1, 3, or 10 predicted results.

---
                                                                                                                                                                                
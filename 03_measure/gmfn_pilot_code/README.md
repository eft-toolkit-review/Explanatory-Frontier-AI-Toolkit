
# CEO Trustworthiness

## Overview
This repository contains the code and data used in the research on CEO trustworthiness prediction using multimodal evidence from CEO videos. 

## Installation
Before running the scripts, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Pandas
- Numpy
- Pickle
- Statsmodels
- Scikit-learn
- EasyDict

You can install these packages using pip:
```bash
pip install torch pandas statsmodels numpy scikit-learn easydict transformers
```

## Dataset
The dataset TBC


### Multimodal Features 
#### Vocal Features
- **Tool**: COVERAP.
- **Features**: 74 dimensions.

#### Visual Features
- **Tool**: OpenFace 2.0.
- **Features**: 247 dimensions.

#### Verbal Features
- **Tool**: BERT.
- **Features**: 768 dimensions.

## Model Description
This codebase implements 14 multimodal models for evaluating CEO trustworthiness, each tailored to different input modality combinations. The core architecture is based on GMFN (Graph Memory Fusion Network), with extended configurations for:

- Unimodal settings (Text / Audio / Image),
- Bimodal combinations (e.g., Text + Audio),
- Trimodal fusion (Text + Audio + Image),
- Ablation studies to investigate the effect of individual modules within GMFN (e.g., temporal reasoning, cross-modal interaction, dynamic fusion graph).

### 1. Unimodal Models (3 Models)

| ID  | File Name                                      | Description                          |
|-----|-----------------------------------------------|--------------------------------------|
| A1  | `A1_GMFN_l_run_evaluation_pretrained_model.py`  | GMFN using **Text only**             |
| A2  | `A2_GMFN_a_run_evaluation_pretrained_model.py`  | GMFN using **Audio only**            |
| A3  | `A3_GMFN_i_run_evaluation_pretrained_model.py`  | GMFN using **Image only**            |

Serve as unimodal baselines to evaluate the individual contribution of each modality.

### 2. Bimodal Models (3 Models)

| ID  | File Name                                      | Description                                |
|-----|-----------------------------------------------|--------------------------------------------|
| A4  | `A4_GMFN_la_run_evaluation_pretrained_model.py`  | GMFN using **Text + Audio**                |
| A5  | `A5_GMFN_li_run_evaluation_pretrained_model.py`  | GMFN using **Text + Image**                |
| A6  | `A6_GMFN_ai_run_evaluation_pretrained_model.py`  | GMFN using **Audio + Image**               |

Compare performance across different modality combinations and investigate multimodal complementarity.

### 3. Trimodal Models and Ablation Studies (8 Models)

| ID   | File Name                                         | Description |
|------|--------------------------------------------------|-------------|
| A7   | `A7_GMFN_run_evaluation_pretrained_model.py`     | GMFN with **Text + Audio + Image** (full fusion) |
| A8   | `A8_GMFN_noM_run_evaluation_pretrained_model.py` | GMFN **without Cross-modal Interaction** module |
| A9   | `A9_GMFN_noG_run_evaluation_pretrained_model.py` | GMFN **without Dynamic Fusion Graph (DFG)** |
| A10  | `A10_GMFN_noW_run_evaluation_pretrained_model.py`| GMFN **without temporal features**, using only static mean features |
| A11  | `A11_EF_LSTM_run_evaluation_pretrained_model.py` | Baseline: Early Fusion LSTM with all three modalities |
| A12  | `A12_TFN_run_evaluation_pretrained_model.py`     | Baseline: TFN (Tensor Fusion Network) |
| A13  | `A13_LMF_run_evaluation_pretrained_model.py`     | Baseline: LMF (Low-rank Multimodal Fusion) |
| A14  | `A14_MFN_run_evaluation_pretrained_model.py`     | Baseline: MFN (Memory Fusion Network) |

Perform comprehensive ablation studies to verify the effectiveness of GMFN components and compare against mainstream multimodal fusion baselines.
Each model is based on pretrained models and demonstrates the prediction results for different data modalities. The models are specifically tailored for analyzing CEO trustworthiness using multimodal data.

## Evaluation Metrics
The code includes metrics for evaluating the model performance:

- Acc2
- F1 Score
- MAE
- Corr

## Pilot Evaluation Results

Evaluation results for all 14 models on the test dataset:

| Model | Acc | F1_score | MAE | Corr |
|---|---:|---:|---:|---:|
| A1_GMFN_l | 0.6283 | 0.5873 | 0.7518 | 0.1839 |
| A2_GMFN_a | 0.6460 | 0.6409 | 0.7484 | 0.2379 |
| A3_GMFN_i | 0.6637 | 0.6589 | 0.7461 | 0.2797 |
| A4_GMFN_la | 0.6903 | 0.6818 | 0.7457 | 0.4568 |
| A5_GMFN_li | 0.7080 | 0.6980 | 0.7438 | 0.4724 |
| A6_GMFN_ai | 0.7434 | 0.7423 | 0.7374 | 0.4897 |
| A7_GMFN | 0.9381 | 0.9381 | 0.4619 | 0.8339 |
| A8_GMFN_noM | 0.8407 | 0.8407 | 0.7307 | 0.6122 |
| A9_GMFN_noG | 0.8850 | 0.8848 | 0.7098 | 0.6955 |
| A10_GMFN_noW | 0.8673 | 0.8673 | 0.7275 | 0.6728 |
| A11_EF_LSTM | 0.9027 | 0.9025 | 0.6216 | 0.7010 |
| A12_TFN | 0.8592 | 0.8581 | 0.6805 | 0.5767 |
| A13_LMF | 0.8732 | 0.8726 | 0.6552 | 0.6154 |
| A14_MFN | 0.8873 | 0.8872 | 0.5925 | 0.6992 |

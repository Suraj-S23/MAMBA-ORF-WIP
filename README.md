# ORF Prediction Project

This project aims to develop and evaluate machine learning models for predicting Open Reading Frames (ORFs) in DNA sequences, particularly focusing on archaeal genomes.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Getting Started](#getting-started)
4. [Running the Project](#running-the-project)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Project Overview

Our project involves developing and comparing different approaches for predicting ORFs in DNA sequences. We are implementing several components:

- A PyTorch-based Mamba model adaptation for archaeal genomes
- A simple baseline model for comparison purposes
- Integration with Prodigal for ORF prediction
- Custom data loading and preprocessing pipelines
- Evaluation scripts for assessing model performance

We aim to improve accuracy, especially for smaller ORFs, and explore the generalization for archaeal genomes.


## Getting Started

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/orf-prediction.git
   cd orf-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

To run the project locally:

1. Ensure you have the necessary dependencies installed.
2. Run the main script:
   ```
   python mamba_model/archaea_mamba.py --config configs/archaea_config.yaml
   ```

## Evaluation Metrics

We evaluate our models based on several metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)

These metrics are calculated for both all ORFs and small ORFs (<150 bases).


## Acknowledgments

We acknowledge the support of the Machine Learning Lab, Albert-Ludwigs University, Freiburg, for this research.

Suraj Subramanian
```

# Statistical Methods in AI — Data Analysis & Machine Learning Portfolio

A curated portfolio of my work in data analysis, statistical modeling, and machine learning across structured, image, text, and temporal datasets.

## Overview

This repository showcases end-to-end workflows for:
- exploratory data analysis (EDA)
- data cleaning and preprocessing
- feature engineering
- statistical modeling
- clustering and dimensionality reduction
- time-series forecasting
- anomaly detection
- deep learning experiments

The work spans tabular, image, text, and sequence data, with a strong focus on model evaluation, visualization, and extracting interpretable insights from noisy or high-dimensional datasets.

## What this repo demonstrates

- **Structured data analysis:** grouped summaries, distribution analysis, correlation heatmaps, sampling studies, k-NN, and regularized regression
- **Unsupervised learning:** K-Means, Gaussian Mixture Models (EM), clustering diagnostics, and transformation-driven structure recovery
- **Dimensionality reduction:** PCA from scratch, explained-variance analysis, and dimensionality–accuracy trade-off studies
- **Forecasting and anomaly detection:** leakage-free temporal pipelines, MAE/RMSE-based evaluation, KDE, and reconstruction-error-based anomaly detection
- **Deep learning:** MLPs, autoencoders, CNNs, VAEs, and transformer-based experiments

## Featured Highlights

### 1) Structured Data Analysis and Statistical Modeling
- analyzed structured datasets using grouped statistics, heatmaps, and conditional visualizations
- compared sampling strategies and regularized regression models
- evaluated predictive performance with validation-based model selection

### 2) Clustering and High-Dimensional Data Analysis
- implemented K-Means, GMM, and PCA pipelines
- analyzed cluster quality using silhouette score, BIC, and convergence behavior
- investigated feature distortions and designed preprocessing steps to improve clustering separability

### 3) Time-Series and Anomaly Detection
- built preprocessing pipelines for temporal data with leakage-free splits
- compared baseline and neural forecasting approaches with horizon-wise error analysis
- used reconstruction error and KDE-based methods for anomaly / foreground detection

## Best Places to Start

If you're reviewing this repo quickly, these sections are the most representative:

| Folder | Focus | Why it matters |
|---|---|---|
| `assignment-1/` | Structured data analysis, sampling, regression, k-NN | Strongest tabular analytics work |
| `assignment-2/` | Clustering, GMM, PCA, transformed-data analysis | Best unsupervised learning and diagnostics |
| `assignment-5/` | KDE, forecasting, anomaly detection, VAE | Strongest real-world analytics flavor |

## Representative Outputs

Add 3 screenshots / plots here from your notebooks:

### Structured Data EDA
![EDA Example](figures/readme/eda_summary.png)

### Clustering / PCA Analysis
![Clustering Example](figures/readme/clustering_diagnostics.png)

### Forecasting / Anomaly Detection
![Forecasting Example](figures/readme/forecast_or_anomaly.png)

## Tech Stack

**Languages:** Python  
**Libraries:** NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib, Seaborn  
**Workflow:** Jupyter Notebook, model evaluation, visualization, experiment analysis

## Repository Structure

- `assignment-1/` — structured tabular data analysis and predictive modeling
- `assignment-2/` — clustering, GMM, PCA, image segmentation, transformed-data analysis
- `assignment-3/` — MLPs from scratch, feature mappings, reconstruction, autoencoders
- `assignment-4/` — CNNs, colorization, trees, random forests
- `assignment-5/` — KDE, time-series forecasting, recurrence discovery, VAE

## How I approach data problems

1. understand the dataset and its failure modes  
2. clean and transform the data  
3. build baseline and advanced models  
4. evaluate with meaningful metrics  
5. visualize results and extract insights  

## Notes

This repository is organized by assignment, but each folder contains self-contained data science workflows involving preprocessing, modeling, evaluation, and interpretation.

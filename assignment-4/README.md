# Assignment 4 — CNNs, Multi-Task Learning, and Interpretable Tree-Based Models

## Overview
This section focuses on supervised learning across image and text data, including convolutional neural networks, multi-task learning, image colourization, and interpretable tree-based methods.

## What it covers
- multi-task CNNs for joint classification and regression
- image colourization with encoder-decoder CNNs
- hyperparameter sweeps and experiment tracking
- decision trees from scratch
- random forests for sentiment classification
- feature importance and model comparison

## Core problems explored
### 1. Multi-task CNN on Fashion-MNIST
Built a shared CNN backbone with separate heads for:
- image classification
- continuous-value regression

Analyzed the effect of:
- joint loss weighting
- optimizer choice
- dropout / batch size / architecture depth
- trade-offs between accuracy and regression error

Tracked experiments using:
- validation accuracy
- MAE
- RMSE
- loss curves across multiple runs

### 2. Image colourization with CNNs
Built an encoder-decoder pipeline for colourizing grayscale CIFAR-10 images by predicting discrete colour clusters.

Key analysis included:
- train / validation loss tracking
- hyperparameter sweeps across learning rate, batch size, filter counts, and kernel sizes
- qualitative comparison of input / predicted / ground-truth outputs
- architecture-level reasoning about learned upsampling and feature extraction

### 3. Decision Trees and Random Forests
Implemented and evaluated interpretable models on bag-of-words sentiment data.

This included:
- decision tree logic from scratch
- grid search for max depth and leaf size
- random forest tuning
- feature-importance analysis
- comparison of simple vs tuned models using balanced accuracy

## Data-science skills demonstrated
- supervised learning across image and text data
- experiment tracking and hyperparameter tuning
- multi-objective model evaluation
- interpretable modeling with trees and forests
- feature importance analysis
- model comparison using validation and test metrics

## Key outputs
- loss and metric curves across CNN runs
- validation / test comparison tables
- feature map visualizations
- grayscale vs predicted-colour vs ground-truth image panels
- tree / forest model comparison tables
- top predictive features from sentiment models

## Why this matters
This folder reflects my ability to work across different data modalities, compare model families fairly, tune models systematically, and communicate results using both metrics and interpretable diagnostics.

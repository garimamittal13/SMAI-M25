# Assignment 3 — Neural Networks from Scratch, Feature Mappings, and Anomaly Detection

## Overview
This section focuses on building neural network pipelines from first principles and analyzing how feature representations affect learning, reconstruction quality, and anomaly detection performance.

## What it covers
- custom multilayer perceptron (MLP) implementation from scratch
- gradient checking and XOR sanity validation
- image reconstruction from spatial coordinates
- comparison of raw, polynomial, and Fourier feature mappings
- reconstruction under increasing blur
- autoencoder-based image reconstruction
- anomaly detection using reconstruction error

## Core problems explored
### 1. Learning decision boundaries with custom MLPs
Implemented a neural network from scratch with:
- linear layers
- ReLU / Tanh / Sigmoid / Identity activations
- BCE and MSE losses
- forward / backward propagation
- gradient accumulation and early stopping

Used the model to learn complex spatial decision boundaries and evaluated the effect of:
- depth
- width
- activation choice
- learning rate / batch size / convergence behavior

### 2. Feature mappings for image reconstruction
Compared how different input representations affect reconstruction quality:
- raw coordinates
- polynomial feature expansions
- Fourier feature mappings

Analyzed:
- convergence speed
- final reconstruction loss
- effect of blur on reconstruction quality
- parameter-efficiency trade-offs across feature mappings

### 3. Autoencoders and anomaly detection
Built deep autoencoders for:
- image reconstruction on MNIST
- anomaly detection on face-image data using reconstruction error

Evaluated anomaly detection performance using:
- AUC
- Precision
- Recall
- F1-score
- ROC and PR analysis

## Data-science skills demonstrated
- neural networks from scratch
- gradient verification and debugging
- representation learning
- reconstruction-error-based anomaly detection
- controlled model comparison
- visualization-driven interpretation of model behavior

## Key outputs
- training loss vs samples / epochs
- reconstruction-quality comparisons across feature mappings
- blur-vs-loss analysis
- original vs reconstructed image comparisons
- ROC / PR curves for anomaly detection
- bottleneck-dimension sensitivity analysis

## Why this matters
This folder demonstrates how I approach modeling problems beyond just using libraries: building the learning pipeline, validating correctness, comparing representations, and using quantitative as well as visual diagnostics to interpret outcomes.

# Placement Predictor 🎯

## Overview
Placement Predictor is a modular machine learning application written in **R** that forecasts student placement outcomes using historical data.  
It consists of two main components:
1. **Prediction Module** – Trains and applies an ML model to predict placement outcomes.
2. **Model Comparison Module** – Evaluates multiple algorithms to identify the highest-performing model.

This design enables flexibility, scalability, and easy integration into enterprise AI workflows.

---

## Features
### 1. Prediction Module
- Loads preprocessed student placement data.
- Trains the chosen model (default: Random Forest Classifier).
- Generates predictions for new student data.
- Outputs results in a structured, easy-to-read format.

### 2. Model Comparison Module
- Benchmarks multiple models:
  - Logistic Regression (`glm`)
  - Decision Tree (`rpart`)
  - Random Forest (`randomForest`)
  - Support Vector Machine (`e1071`)
- Calculates and displays accuracy for each model.
- Visualizes model performance with bar plots.

---

## Tech Stack
- **Language**: R  
- **Libraries**: `caret`, `rpart`, `randomForest`, `e1071`, `ggplot2`
- **Concepts**: Feature scaling, missing value handling, classification, model benchmarking, visualization

---

## AI Relevance
- **Enterprise-Ready**: Modular R scripts can be integrated into dashboards, APIs, or enterprise data workflows.
- **Scalable**: Easily add new models or update datasets.
- **Explainable AI**: Accuracy comparison module helps stakeholders select the most reliable model.

---

## Results
- Best Model: **Random Forest Classifier**
- Accuracy: **85%**



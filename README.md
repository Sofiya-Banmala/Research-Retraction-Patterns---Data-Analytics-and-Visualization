# Uncovering Insights: A Data-Driven Examination of Research Retraction Patterns

## Overview

This repository contains the codebase, data preprocessing steps, visualizations, and analytical outcomes developed for a research project submitted for **PRT564 – Data Analytics and Visualisation** at **Charles Darwin University**. The project investigates the patterns, causes, and implications of retracted academic research articles using data analytics and machine learning techniques.

---

## Objective

The principal aim of this study is to explore the trends and reasoning behind scientific paper retractions. Through computational methods such as clustering, regression, and classification, we aim to build a foundational understanding of the dynamics behind retractions in academia, especially within biomedical and life sciences literature.

---

## Authors

**Group 9 (Sydney Campus)**  
- Andrea Vijeetha Marlene Vijay (S371784)  
- Manisha KC (S372043)  
- Manoj Poudel (S372470)  
- Sofiya Banmala (S372189)

**Supervisor:** Prof. Niusha Shafi Abady

---

## Methodology

This project follows a structured data analytics pipeline:

### 1. Data Preprocessing
- Source: Retraction Watch dataset (35,000+ records)
- Null value treatment, feature selection, and cleaning using Pandas

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics
- Retraction trends over time
- Retractions by country, journal, and subject area
- Word cloud and correlation heatmaps

### 3. Dimensionality Reduction & Clustering
- **PCA (Principal Component Analysis)** to reduce dimensionality
- **K-Means Clustering** to group retractions by shared characteristics
- **Isolation Forest** to identify outliers (e.g., suspicious retractions)

### 4. Predictive Modeling
- **Multiple Linear Regression** to model citation count
- **Classification Models:**
  - Random Forest Classifier (accuracy: ~95%)
  - Categorical Naive Bayes (accuracy: ~88%)
  - Support Vector Machine (accuracy: ~50%)

### 5. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, R², and MSE

---

## Technologies Used

- **Language**: Python 3.x  
- **Libraries**:  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `scikit-learn`, `statsmodels`  
- **Tools**: Jupyter Notebook, Git

---

## Key Findings

- The majority of retractions occurred in **medical and biological sciences**.
- **USA** and **China** reported the highest number of retractions.
- Common causes: **Data fabrication, plagiarism, unethical research, peer review manipulation**
- Predictive models suggest potential for early detection of high-risk publications.
- Regression models revealed statistically significant predictors but low explanatory power (R² = 0.04).

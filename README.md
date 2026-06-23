# Transaction Fraud Detection & Prevention Pipeline
<img width="958" height="727" alt="image" src="https://github.com/user-attachments/assets/570ef79b-9d31-41e4-a992-0a54b77c4ed0" />
<img width="954" height="737" alt="image" src="https://github.com/user-attachments/assets/bd0e7c05-b3c3-4167-88c9-b76b04f05594" />

An end-to-end Machine Learning pipeline and real-time operations dashboard designed to intercept transactional fraud and prevent an estimated KES 1.7 Billion in annual capital leakage.

This system replaces static, easily bypassed rules-based fraud flags with a machine learning engine calibrated to detect complex behavioral patterns, such as impossible travel anomalies, transaction velocity spikes, and extreme deviations from historical customer benchmarks.

## Business Problem

Legacy transaction monitoring systems rely on fixed thresholds (e.g., flagging any transaction over KES 100,000). Modern fraud rings easily circumvent these filters by executing low-velocity micro-transations or routing stolen funds through atypical channels. The financial damage goes beyond direct loss of funds; high false-positive rates degrade customer trust and overwhelm operational risk units with manual investigation pipelines.

## Solution Architecture

The pipeline consists of three core components:

**transaction_generator.py (Behavioral Ledger Synthesizer)**

* Simulates a transaction ledger of 50,000 entries.

* Models authentic daily transaction baselines across major channels (Mobile App, Agent Cash Out, ATM, Web, and POS).

* Programmatically injects highly imbalanced (0.6%) fraud cases using specific risk profiles: Impossible Travel, Micro-transaction Velocity Spikes, and Midnight Behavioral Deviations.

**fraud_detector.py (Class-Weighted Random Forest Engine)**

* Implements custom feature engineering, converting absolute figures into behavioral ratios (e.g., Transaction Amount relative to Historical Average).

* Splits and stratifies data to manage extreme class imbalance.

* Leverages a Random Forest Classifier trained with balanced class weights to maximize model sensitivity (Recall).

* Scores the entire transaction database, exporting the risk allocation to scored_transactions.csv.

## fraud_dashboard.py (Fraud Operations Command Center)

* A Streamlit interface designed for forensic risk analysts and security officers.

* Translates mathematical classification metrics into operational business outcomes: Prevented Financial Damage (KES Millions), Fraud Interception Rate, and Analyst Workload Ratio (managing false alarms to actual intercepts).

* Offers interactive risk filtering and an individual forensic case fact sheet.

## Performance Metrics Achieved

Based on the optimized model run, the system demonstrates the following risk mitigation metrics:

* Model ROC-AUC: 0.9582 (demonstrating exceptionally high diagnostic discrimination capacity).

* Interception Rate (Recall): 89.09% (capturing nearly 9 out of 10 fraud attempts before settlement).

* Analyst Workload Ratio: 14.5:1 (a highly efficient baseline for financial fraud operations, ensuring risk analysts resolve 14.5 triggered alerts to confirm and block 1 high-priority fraud case).

## Technical Stack

Runtime Environment: Python 3.10+

Analytical Engines: Scikit-Learn (Random Forest Classification), Pandas, NumPy

Visualization Layer: Streamlit, Plotly (Graph Objects, Express)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings

# Suppress sklearn warnings for clean console output
warnings.filterwarnings('ignore')

def run_fraud_detector():
    print("Loading historical transaction database...")
    try:
        df = pd.read_csv("historical_transactions.csv")
    except FileNotFoundError:
        print("SYSTEM ERROR: historical_transactions.csv not found. Please run transaction_generator.py first.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ADVANCED FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────────
    print("Executing feature engineering pipeline...")
    
    # Feature 1: Deviation Ratio (How many times larger is this transaction than their normal average?)
    df['Amount_to_Historical_Ratio'] = df['Amount_KES'] / df['Historical_Average_KES']
    
    # Feature 2: Is the transaction occurring in a high-risk location relative to home base?
    # (Since Nairobi/Mombasa represent local nodes, London/Dubai are flagged as international/high-risk nodes)
    df['Is_High_Risk_Location'] = df['Location'].apply(lambda x: 1 if x in ['London', 'Dubai'] else 0)
    
    # 3. Categorical Encoding
    # Encode 'Channel' and 'Location' into numeric dummy features
    categorical_features = ['Channel', 'Location']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Define features to train the machine learning model
    exclude_cols = ['Transaction_ID', 'Customer_ID', 'Timestamp', 'Is_Fraud', 'Fraud_Type']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X = df_encoded[feature_cols]
    y = df_encoded['Is_Fraud']
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. TRAIN/TEST SPLIT (STRATIFIED TO PRESERVE IMBALANCE RATIO)
    # ─────────────────────────────────────────────────────────────────────────
    # Stratified split ensures both train and test sets contain the exact same 0.6% proportion of fraud
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. MACHINE LEARNING MODEL TRAINING (CLASS-WEIGHT BALANCED)
    # ─────────────────────────────────────────────────────────────────────────
    print("Training Random Forest Classifier (Optimized for Class Imbalance)...")
    
    # We use class_weight='balanced' so the model penalizes missing a fraud case much more heavily
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. MODEL EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\nEvaluating model performance against testing baseline...")
    
    # Generate probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC-AUC Score (World-Class threshold is >0.90)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Model ROC-AUC Score: {roc_auc:.4f}")
    
    # Dynamic Threshold Calibration
    # In highly imbalanced fraud setups, a standard 0.5 threshold fails.
    # Setting threshold to 0.3 captures more risk cases (higher recall) while protecting operational efficiency.
    decision_threshold = 0.30
    y_pred_custom = (y_pred_prob >= decision_threshold).astype(int)
    
    print(f"\nClassification Metrics (Risk Threshold: {decision_threshold}):")
    print(classification_report(y_test, y_pred_custom, target_names=["Legitimate", "Fraud"]))
    
    # Extract Confusion Matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_custom).ravel()
    print("Confusion Matrix Overview:")
    print(f"  - Correctly Blocked Fraud Cases (True Positives): {tp}")
    print(f"  - Legitimate Transactions Passed (True Negatives): {tn}")
    print(f"  - Fraud Cases Missed (False Negatives): {fn}")
    print(f"  - False Positive Alarms Triggered: {fp}")
    
    # Calculate operational impact metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nPrecision: {precision:.2%} (Percentage of flags that are actual fraud)")
    print(f"Recall: {recall:.2%} (Percentage of total fraud successfully intercepted)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. FULL LEDGER SCORING
    # ─────────────────────────────────────────────────────────────────────────
    print("\nScoring total historical ledger...")
    
    # Append calculated probabilities and baseline features to original dataframe
    df['Fraud_Probability'] = model.predict_proba(X)[:, 1]
    
    # Define risk categorization based on calibrated probability bands
    def determine_risk_tier(prob):
        if prob >= 0.70:
            return "High Risk (Critical)"
        elif prob >= 0.30:
            return "Medium Risk (Suspicious)"
        else:
            return "Low Risk (Verified)"
            
    df['Risk_Tier'] = df['Fraud_Probability'].apply(determine_risk_tier)
    
    # Export fully scored transactions for the Dashboard Command Center
    output_filename = "scored_transactions.csv"
    df.to_csv(output_filename, index=False)
    print(f"Scored ledger successfully written to: {output_filename}")

if __name__ == "__main__":
    run_fraud_detector()
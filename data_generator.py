import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_transaction_data(num_transactions=50000):
    print("Initializing Transaction Data Synthesizer...")
    np.random.seed(42)
    random.seed(42)

    # Base parameters
    channels = ["Mobile App", "Agent Cash Out", "ATM", "Web Purchase", "POS Terminal"]
    locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "London", "Dubai"]
    location_weights = [0.60, 0.15, 0.10, 0.08, 0.05, 0.01, 0.01]

    # Generate customer base profiles (to map typical historical amounts)
    customer_profiles = {}
    for i in range(1, 2501):  # 2,500 active customers
        customer_profiles[f"CUST-{str(i).zfill(5)}"] = {
            "avg_amount": np.random.lognormal(mean=7.5, sigma=0.8),  # Median around KES 1,800
            "primary_location": np.random.choice(locations, p=location_weights)
        }

    transactions = []
    base_time = datetime.now() - timedelta(days=60)

    # Generate sequential chronological transactions
    for i in range(num_transactions):
        trans_id = f"TXN-{str(i + 1).zfill(6)}"
        cust_id = f"CUST-{str(random.randint(1, 2500)).zfill(5)}"
        profile = customer_profiles[cust_id]

        # Time step forward
        base_time += timedelta(seconds=random.randint(10, 300))
        timestamp = base_time

        # Select standard metrics
        channel = np.random.choice(channels, p=[0.45, 0.25, 0.15, 0.10, 0.05])
        location = profile["primary_location"] if random.random() < 0.90 else np.random.choice(locations)
        
        # Calculate standard transaction amount with noise
        amount = np.random.normal(loc=profile["avg_amount"], scale=profile["avg_amount"] * 0.3)
        amount = max(round(amount, 2), 100.0)  # Min transaction KES 100

        # Behavioral markers
        hour = timestamp.hour
        is_night = 1 if (hour >= 23 or hour <= 5) else 0

        # Initialize labels
        is_fraud = 0
        fraud_type = "Legitimate"

        # Inject Simulated Fraud Scenarios (target roughly 0.6% total fraud)
        trigger = random.random()
        
        if trigger < 0.006:
            is_fraud = 1
            fraud_type_choice = random.choice(["Impossible Travel", "Velocity Spike", "Value Deviation"])
            
            if fraud_type_choice == "Impossible Travel":
                fraud_type = "Impossible Travel"
                # Override location to something distant (e.g. London or Dubai)
                location = "London" if profile["primary_location"] != "London" else "Nairobi"
                amount = round(np.random.uniform(5000, 45000), 2)
                
            elif fraud_type_choice == "Velocity Spike":
                fraud_type = "Velocity Spike"
                # Micro-amount velocity spamting
                amount = round(np.random.uniform(500, 2000), 2)
                # Ensure it looks like midnight/abnormal time
                timestamp = timestamp.replace(hour=random.choice([1, 2, 3, 4]))
                
            elif fraud_type_choice == "Value Deviation":
                fraud_type = "Value Deviation"
                # Massive transfer relative to historic profile
                amount = round(profile["avg_amount"] * np.random.uniform(15, 30), 2)
                timestamp = timestamp.replace(hour=random.choice([0, 1, 2, 3, 4, 5]))
                channel = "Web Purchase"

        transactions.append({
            "Transaction_ID": trans_id,
            "Customer_ID": cust_id,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Amount_KES": amount,
            "Channel": channel,
            "Location": location,
            "Hour_of_Day": hour,
            "Is_Night": is_night,
            "Historical_Average_KES": round(profile["avg_amount"], 2),
            "Is_Fraud": is_fraud,
            "Fraud_Type": fraud_type
        })

    # Compile and export
    df = pd.DataFrame(transactions)
    output_file = "historical_transactions.csv"
    df.to_csv(output_file, index=False)

    print("\n--- SYNTHESIS COMPLETE ---")
    print(f"Total Transactions Generated: {len(df)}")
    print(f"Total Fraud Injected: {len(df[df['Is_Fraud'] == 1])} ({len(df[df['Is_Fraud'] == 1]) / len(df) * 100:.2f}%)")
    print(f"Data successfully saved to: {output_file}")
    print("\nFraud Type Distribution:")
    print(df[df['Is_Fraud'] == 1]['Fraud_Type'].value_counts().to_string())

if __name__ == "__main__":
    generate_transaction_data()
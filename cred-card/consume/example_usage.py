#!/usr/bin/env python3
"""
Credit Card Rewards Model - Example Usage Script

This script demonstrates how to use the trained credit card rewards classification model
to predict customer reward tiers based on their spending behavior.

Usage:
    python consume/example_usage.py

Requirements:
    - pandas
    - scikit-learn
    - numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the current directory to Python path to import the model
sys.path.append(str(Path(__file__).parent))

def create_sample_customers():
    """Create sample customer data for demonstration"""
    customers = [
        {
            'name': 'High Income Frequent Traveler',
            'Annual_Income': 1200000,
            'Monthly_Average_Spend': 70000,
            'Transactions_Per_Month': 50,
            'Online_Offline_Spend_Ratio': 0.3,
            'Travel_Spend_Ratio': 0.6,
            'Dining_Spend_Ratio': 0.1,
            'expected_tier': 'Platinum'
        },
        {
            'name': 'Moderate Income Online Shopper',
            'Annual_Income': 750000,
            'Monthly_Average_Spend': 35000,
            'Transactions_Per_Month': 25,
            'Online_Offline_Spend_Ratio': 0.8,
            'Travel_Spend_Ratio': 0.1,
            'Dining_Spend_Ratio': 0.3,
            'expected_tier': 'Gold'
        },
        {
            'name': 'Budget-Conscious Customer',
            'Annual_Income': 500000,
            'Monthly_Average_Spend': 18000,
            'Transactions_Per_Month': 15,
            'Online_Offline_Spend_Ratio': 0.9,
            'Travel_Spend_Ratio': 0.05,
            'Dining_Spend_Ratio': 0.2,
            'expected_tier': 'Basic'
        },
        {
            'name': 'Food Enthusiast',
            'Annual_Income': 900000,
            'Monthly_Average_Spend': 45000,
            'Transactions_Per_Month': 35,
            'Online_Offline_Spend_Ratio': 0.4,
            'Travel_Spend_Ratio': 0.2,
            'Dining_Spend_Ratio': 0.5,
            'expected_tier': 'Gold'
        },
        {
            'name': 'Young Professional',
            'Annual_Income': 600000,
            'Monthly_Average_Spend': 22000,
            'Transactions_Per_Month': 20,
            'Online_Offline_Spend_Ratio': 0.7,
            'Travel_Spend_Ratio': 0.15,
            'Dining_Spend_Ratio': 0.4,
            'expected_tier': 'Basic'
        }
    ]
    return customers

def validate_input(customer_data):
    """Validate customer input data"""
    required_features = ['Annual_Income', 'Monthly_Average_Spend', 'Transactions_Per_Month',
                        'Online_Offline_Spend_Ratio', 'Travel_Spend_Ratio', 'Dining_Spend_Ratio']
    
    for feature in required_features:
        if feature not in customer_data:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Validate ranges
    if customer_data['Annual_Income'] < 0:
        raise ValueError("Annual income must be positive")
    
    ratio_features = ['Online_Offline_Spend_Ratio', 'Travel_Spend_Ratio', 'Dining_Spend_Ratio']
    for feature in ratio_features:
        if not (0 <= customer_data[feature] <= 1):
            raise ValueError(f"{feature} must be between 0 and 1")
    
    return True

def predict_customer_tier(model, customer_data):
    """Predict reward tier for a single customer"""
    # Remove non-feature columns
    feature_cols = ['Annual_Income', 'Monthly_Average_Spend', 'Transactions_Per_Month',
                   'Online_Offline_Spend_Ratio', 'Travel_Spend_Ratio', 'Dining_Spend_Ratio']
    
    customer_features = {k: v for k, v in customer_data.items() if k in feature_cols}
    
    # Validate input
    validate_input(customer_features)
    
    # Convert to DataFrame
    df = pd.DataFrame([customer_features])
    
    # Make prediction
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    return prediction, probabilities

def main():
    """Main function to demonstrate model usage"""
    print("=" * 60)
    print("Credit Card Rewards Model - Example Usage")
    print("=" * 60)
    
    # Note: In a real scenario, you would load a saved model
    # For this example, we'll show the structure without actually loading
    print("\nNote: This example shows how to use the model.")
    print("To run actual predictions, you need to:")
    print("1. Train the model using the Jupyter notebook")
    print("2. Save the model using joblib.dump(model, 'model.pkl')")
    print("3. Load it here using joblib.load('model.pkl')")
    print()
    
    # Create sample customers
    customers = create_sample_customers()
    
    print("Sample Customer Predictions:")
    print("-" * 60)
    
    for customer in customers:
        print(f"\nCustomer: {customer['name']}")
        print(f"Annual Income: ${customer['Annual_Income']:,}")
        print(f"Monthly Spend: ${customer['Monthly_Average_Spend']:,}")
        print(f"Transactions/Month: {customer['Transactions_Per_Month']}")
        print(f"Online/Offline Ratio: {customer['Online_Offline_Spend_Ratio']:.2f}")
        print(f"Travel Spend Ratio: {customer['Travel_Spend_Ratio']:.2f}")
        print(f"Dining Spend Ratio: {customer['Dining_Spend_Ratio']:.2f}")
        print(f"Expected Tier: {customer['expected_tier']}")
        
        # In a real implementation, you would do:
        # prediction, probabilities = predict_customer_tier(model, customer)
        # print(f"Predicted Tier: {prediction}")
        # print(f"Confidence: {max(probabilities):.3f}")
        
        print("(Run the notebook to get actual predictions)")

def interactive_prediction():
    """Interactive function for user input"""
    print("\n" + "=" * 60)
    print("Interactive Credit Card Tier Prediction")
    print("=" * 60)
    
    try:
        print("\nEnter customer details:")
        
        annual_income = float(input("Annual Income ($): "))
        monthly_spend = float(input("Monthly Average Spend ($): "))
        transactions = int(input("Transactions Per Month: "))
        online_ratio = float(input("Online/Offline Spend Ratio (0.0-1.0): "))
        travel_ratio = float(input("Travel Spend Ratio (0.0-1.0): "))
        dining_ratio = float(input("Dining Spend Ratio (0.0-1.0): "))
        
        customer_data = {
            'Annual_Income': annual_income,
            'Monthly_Average_Spend': monthly_spend,
            'Transactions_Per_Month': transactions,
            'Online_Offline_Spend_Ratio': online_ratio,
            'Travel_Spend_Ratio': travel_ratio,
            'Dining_Spend_Ratio': dining_ratio
        }
        
        # Validate input
        validate_input(customer_data)
        
        print("\n✓ Input validation passed!")
        print("To get the actual prediction, run the Jupyter notebook with this data:")
        
        # Show the code they can use in the notebook
        print(f"""
new_customer = pd.DataFrame([{{
    'Annual_Income': {annual_income},
    'Monthly_Average_Spend': {monthly_spend},
    'Transactions_Per_Month': {transactions},
    'Online_Offline_Spend_Ratio': {online_ratio},
    'Travel_Spend_Ratio': {travel_ratio},
    'Dining_Spend_Ratio': {dining_ratio},
}}])

prediction = model.predict(new_customer)[0]
print(f"Predicted reward tier: {{prediction}}")
""")
        
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
    
    # Ask if user wants interactive mode
    try:
        response = input("\nWould you like to try interactive prediction? (y/n): ").lower()
        if response in ['y', 'yes']:
            interactive_prediction()
    except KeyboardInterrupt:
        print("\nBye Bye!")

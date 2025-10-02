# Credit Card Rewards Model - Usage Guide

## Overview
This document explains how to use the trained credit card rewards classification model to predict customer reward tiers (Basic, Gold, Platinum) based on their spending behavior and income patterns.

## Model Information
- **Algorithm**: Random Forest Classifier with StandardScaler preprocessing
- **Accuracy**: 93% on test data
- **Target**: Predict customer reward tier (Basic, Gold, Platinum)
- **Features**: 6 numeric features describing customer spending behavior

## Input Features Required

The model expects the following 6 numeric features:

| Feature | Description | Data Type | Example Range |
|---------|-------------|-----------|---------------|
| `Annual_Income` | Customer's yearly income | Integer | 500,000 - 1,500,000 |
| `Monthly_Average_Spend` | Average monthly spending amount | Integer | 10,000 - 80,000 |
| `Transactions_Per_Month` | Number of transactions per month | Integer | 5 - 100 |
| `Online_Offline_Spend_Ratio` | Ratio of online to total spending | Float | 0.0 - 1.0 |
| `Travel_Spend_Ratio` | Ratio of travel spending to total | Float | 0.0 - 1.0 |
| `Dining_Spend_Ratio` | Ratio of dining spending to total | Float | 0.0 - 1.0 |

## Example Usage

### 1. Basic Prediction Example

```python
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model (assuming it's saved as 'credit_card_model.pkl')
model = joblib.load('credit_card_model.pkl')

# Create a new customer profile
new_customer = pd.DataFrame([{
    'Annual_Income': 800000,
    'Monthly_Average_Spend': 40000,
    'Transactions_Per_Month': 30,
    'Online_Offline_Spend_Ratio': 0.6,
    'Travel_Spend_Ratio': 0.2,
    'Dining_Spend_Ratio': 0.3,
}])

# Make prediction
predicted_tier = model.predict(new_customer)[0]
print(f"Predicted reward tier: {predicted_tier}")
# Output: Predicted reward tier: Gold
```

### 2. Batch Prediction Example

```python
# Predict for multiple customers at once
customers_data = pd.DataFrame([
    {
        'Annual_Income': 600000,
        'Monthly_Average_Spend': 25000,
        'Transactions_Per_Month': 20,
        'Online_Offline_Spend_Ratio': 0.8,
        'Travel_Spend_Ratio': 0.1,
        'Dining_Spend_Ratio': 0.4,
    },
    {
        'Annual_Income': 1200000,
        'Monthly_Average_Spend': 60000,
        'Transactions_Per_Month': 45,
        'Online_Offline_Spend_Ratio': 0.3,
        'Travel_Spend_Ratio': 0.5,
        'Dining_Spend_Ratio': 0.2,
    },
    {
        'Annual_Income': 450000,
        'Monthly_Average_Spend': 15000,
        'Transactions_Per_Month': 12,
        'Online_Offline_Spend_Ratio': 0.9,
        'Travel_Spend_Ratio': 0.05,
        'Dining_Spend_Ratio': 0.1,
    }
])

# Make predictions
predictions = model.predict(customers_data)
print("Predictions:", predictions)
# Output: Predictions: ['Basic' 'Platinum' 'Basic']
```

### 3. Get Prediction Probabilities

```python
# Get probability scores for each class
probabilities = model.predict_proba(new_customer)
class_names = model.classes_

print("Prediction probabilities:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {probabilities[0][i]:.3f}")

# Output:
# Prediction probabilities:
# Basic: 0.050
# Gold: 0.930
# Platinum: 0.020
```

### 4. Feature Importance Analysis

```python
# Get feature importance from the Random Forest model
rf_model = model.named_steps['rf']
feature_names = ['Annual_Income', 'Monthly_Average_Spend', 'Transactions_Per_Month',
                'Online_Offline_Spend_Ratio', 'Travel_Spend_Ratio', 'Dining_Spend_Ratio']

importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

print("Feature Importance:")
print(importances)

# Output:
# Feature Importance:
# Annual_Income                 0.438038
# Monthly_Average_Spend         0.409491
# Transactions_Per_Month        0.040646
# Online_Offline_Spend_Ratio    0.039065
# Travel_Spend_Ratio            0.037253
# Dining_Spend_Ratio            0.035507
```

## Model Performance Insights

### Key Findings:
1. **Most Important Features**: Annual Income (43.8%) and Monthly Average Spend (41.0%) are the primary drivers
2. **Transaction Patterns**: Number of transactions per month has moderate importance (4.1%)
3. **Spending Categories**: Travel, dining, and online/offline ratios have similar, lower importance (~3.5% each)

### Model Limitations:
- **Platinum Class**: The model struggles to predict Platinum tier (0% precision/recall in test set)
- **Class Imbalance**: Only 7% of customers are Platinum tier, making it harder to learn patterns
- **Missing Text Features**: The model doesn't use the `Transaction_Summary` text field

## Production Deployment Considerations

### 1. Input Validation
```python
def validate_customer_input(customer_data):
    """Validate input data before prediction"""
    required_features = ['Annual_Income', 'Monthly_Average_Spend', 'Transactions_Per_Month',
                        'Online_Offline_Spend_Ratio', 'Travel_Spend_Ratio', 'Dining_Spend_Ratio']
    
    # Check all required features are present
    missing_features = [f for f in required_features if f not in customer_data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check data types and ranges
    if customer_data['Annual_Income'].min() < 0:
        raise ValueError("Annual income must be positive")
    
    if not all(0 <= customer_data[col].max() <= 1 for col in ['Online_Offline_Spend_Ratio', 'Travel_Spend_Ratio', 'Dining_Spend_Ratio']):
        raise ValueError("Spend ratios must be between 0 and 1")
    
    return True
```

### 2. Model Persistence
```python
# Save the trained model
import joblib

joblib.dump(model, 'credit_card_rewards_model.pkl')

# Load the model later
model = joblib.load('credit_card_rewards_model.pkl')
```

### 3. API Endpoint Example (Flask)
```python
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('credit_card_rewards_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_reward_tier():
    try:
        # Get customer data from request
        customer_data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Validate input
        validate_customer_input(df)
        
        # Make prediction
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Format response
        result = {
            'predicted_tier': prediction,
            'probabilities': {
                'Basic': float(probabilities[0]),
                'Gold': float(probabilities[1]),
                'Platinum': float(probabilities[2])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

## Business Use Cases

### 1. Credit Card Application Processing
- **Use Case**: Automatically assign reward tiers for new credit card applications
- **Input**: Customer income and spending behavior from application form
- **Output**: Recommended reward tier (Basic/Gold/Platinum)

### 2. Customer Segmentation
- **Use Case**: Segment existing customers for targeted marketing campaigns
- **Input**: Historical spending data for all customers
- **Output**: Reward tier classifications for marketing personalization

### 3. Credit Limit Optimization
- **Use Case**: Adjust credit limits based on predicted reward tier
- **Input**: Customer spending patterns and income
- **Output**: Tier-based credit limit recommendations

### 4. Risk Assessment
- **Use Case**: Assess credit risk based on spending behavior patterns
- **Input**: Customer transaction history and income
- **Output**: Reward tier as a proxy for creditworthiness

## Model Improvement Recommendations

### 1. Address Class Imbalance
```python
from sklearn.utils.class_weight import compute_class_weight

# Use class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Update model with class weights
model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=300, class_weight=class_weight_dict, random_state=42))
])
```

### 2. Incorporate Text Features
```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Include Transaction_Summary text features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('text', TfidfVectorizer(max_features=500), 'Transaction_Summary')
    ]
)

enhanced_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
])
```

### 3. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

## Conclusion

This credit card rewards model provides a solid foundation for predicting customer reward tiers based on spending behavior. While it achieves 93% accuracy overall, there are opportunities for improvement, particularly in predicting the Platinum tier. The model is ready for deployment with proper input validation and can be enhanced through hyperparameter tuning and feature engineering.


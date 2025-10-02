## Credit Card Rewards Classification (Beginner-Friendly)

This mini-project predicts a customer's credit card reward tier (Basic, Gold, Platinum) from their spending behavior.

### What's in this repo
- `data/credit_card_rewards_5000.csv`: the dataset.
- `notebooks/credit_card_rewards_analysis.ipynb`: a Jupyter notebook with EDA, modeling, and evaluation.
- `requirements.txt`: Python packages used.
- `consume/MODEL_USAGE.md`: comprehensive guide on how to use the trained model.
- `consume/example_usage.py`: script demonstrating model usage with sample customers.

### Quick start
1) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Launch Jupyter and open the notebook
```
jupyter lab
```
Then open `notebooks/credit_card_rewards_analysis.ipynb` and select the kernel named "Python (.venv) cred-card". Run cells top-to-bottom.

### What the notebook does (in plain language)
1) Load data and preview columns
   - Columns include income, average monthly spend, number of transactions, and ratios like online/offline, travel, dining.
   - `Reward_Tier` is the label to predict.

2) Quick EDA (exploratory data analysis)
   - Shows dataset size, column types, and missing values.
   - Plots distributions of numeric features and a correlation heatmap.
   - Checks class balance of `Reward_Tier`.

3) Build a simple model pipeline
   - Uses numeric features: income, spend, transactions, and spend ratios.
   - Scales numeric features and trains a Random Forest classifier.

4) Evaluate the model
   - Train/test split to estimate performance on unseen data.
   - Prints accuracy and a classification report (precision/recall per class).
   - Shows a confusion matrix heatmap to see where the model confuses classes.
   - Reports feature importance (from the Random Forest).
   - Optional: cross-validated accuracy for a more robust estimate.

5) Optional: use text from `Transaction_Summary`
   - Adds a second pipeline that turns text into features using TF‑IDF.
   - Combines numeric + text features and trains Logistic Regression.
   - Prints accuracy for this extended approach.

### Interpreting results
- Accuracy: overall fraction of correct predictions.
- Classification report: per-class precision/recall/F1; helpful if classes are imbalanced.
- Confusion matrix: shows which classes get mixed up.
- Feature importances (RF): which numeric features the model relied on more.

### How this could be improved
- Try more models and hyperparameter tuning (e.g., XGBoost/LightGBM, tuned RF/LogReg).
- Use better text processing (stopwords, n‑grams, domain keywords) for `Transaction_Summary`.
- Calibrate probabilities and set thresholds aligned to business goals.
- Monitor performance over time and re‑train as behavior changes.
- Add fairness checks and ensure consistent performance across customer segments.

### Predicting for a new customer (example)
Inside the notebook, you’ll find a small snippet like this:
```
new_customer = pd.DataFrame([
    {
        'Annual_Income': 800000,
        'Monthly_Average_Spend': 40000,
        'Transactions_Per_Month': 30,
        'Online_Offline_Spend_Ratio': 0.6,
        'Travel_Spend_Ratio': 0.2,
        'Dining_Spend_Ratio': 0.3,
    }
])
model.predict(new_customer)[0]
```
This returns the predicted reward tier.

### Using the trained model

After running the notebook and training the model, you can use it to predict reward tiers for new customers:

#### Quick prediction example:
```python
import pandas as pd

# Example customer data
new_customer = pd.DataFrame([{
    'Annual_Income': 800000,
    'Monthly_Average_Spend': 40000,
    'Transactions_Per_Month': 30,
    'Online_Offline_Spend_Ratio': 0.6,
    'Travel_Spend_Ratio': 0.2,
    'Dining_Spend_Ratio': 0.3,
}])

# Predict reward tier
predicted_tier = model.predict(new_customer)[0]
print(f"Predicted reward tier: {predicted_tier}")
```

#### For detailed usage instructions:
- See `consume/MODEL_USAGE.md` for comprehensive documentation
- Run `python consume/example_usage.py` for interactive examples
- The model achieves 93% accuracy on test data

### Learn the math behind the notebook (beginner-friendly)
- Read `docs/beginner_math_guide.md` for step-by-step intuition and formulas used in the notebook (scaling, Random Forests, metrics, TF‑IDF, logistic regression, cross-validation, and more).

### Deeper documentation
- `docs/notebook_walkthrough.md`: line-by-line explanation of the notebook’s code, decisions, and pitfalls to avoid.
- `docs/pipeline_and_inference.md`: how to productionize the pipeline, validate inputs, tune hyperparameters, persist artifacts, and monitor performance.
- `docs/glossary.md`: concise definitions of key terms used across the project.

### Tips for beginners
- Run the notebook one cell at a time; read outputs before moving on.
- If something breaks, check column names and that you're using the right kernel.
- Use the visuals (histograms, heatmap, confusion matrix) to build intuition.



## Glossary (Beginner-Friendly)

### Dataset & Features
- **Feature (Variable)**: A measurable input used by the model (e.g., `Annual_Income`).
- **Label (Target)**: What we want to predict (here: `Reward_Tier`).
- **Numeric Feature**: A number-valued input (income, spend, ratios).
- **Text Feature**: Free‑text input (e.g., `Transaction_Summary`).
- **Train/Test Split**: Split the dataset into parts to train the model and evaluate it on unseen data.

### EDA (Exploratory Data Analysis)
- **Shape**: Number of rows (examples) and columns (features) in the dataset.
- **Dtypes**: Data types (int, float, object/text) for each column.
- **Missing Values**: Empty or null entries that may need cleaning.
- **Distribution**: How values are spread across a feature (shown with histograms).
- **Correlation**: A number from −1 to 1 indicating linear relationship between two numeric features.
- **Class Balance**: How many examples exist for each class in the label.

### Modeling & Pipelines
- **Classifier**: A model that predicts a category (e.g., Basic/Gold/Platinum).
- **Random Forest**: An ensemble of decision trees; good default for tabular data.
- **Logistic Regression**: A linear model for classification; outputs class probabilities.
- **Pipeline**: A chain of steps (e.g., scaling → model) that runs together.
- **Standardization (Scaling)**: Transform features to have mean 0 and standard deviation 1.
- **ColumnTransformer**: Applies different preprocessing to different columns (e.g., scale numbers, TF‑IDF for text).

### Text Features
- **Token**: A unit of text (usually a word or subword).
- **n‑gram**: Sequences of n tokens (e.g., bigram = 2 tokens).
- **TF‑IDF (Term Frequency–Inverse Document Frequency)**: A way to convert text to numeric features by weighing words higher if they are frequent in a document but rare across the corpus.

### Evaluation
- **Accuracy**: Fraction of correct predictions.
- **Precision**: Of the predicted positives for a class, how many are truly that class?
- **Recall (Sensitivity)**: Of the actual positives for a class, how many did we correctly predict?
- **F1 Score**: Harmonic mean of precision and recall; balances both.
- **Confusion Matrix**: Table showing counts of actual vs predicted classes.
- **Cross‑Validation (CV)**: Repeatedly split the data into train/test folds to get a more stable performance estimate.

### Practical Concerns
- **Hyperparameter Tuning**: Searching for the best model settings (e.g., number of trees).
- **Calibration**: Adjust predicted probabilities to better reflect true likelihoods.
- **Class Imbalance**: When some classes have many more examples than others; may need class weights or resampling.
- **Drift**: Data patterns change over time; models should be monitored and retrained.
- **Fairness**: Ensuring performance is equitable across different customer groups.

### Inference & Deployment
- **Inference**: Using a trained model to make predictions on new data.
- **Serialization (Model Saving)**: Saving a trained model (e.g., via `joblib`) for later use.
- **Input Validation**: Checking that incoming data has expected types, ranges, and columns.



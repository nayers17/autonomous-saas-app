import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.datasets import make_classification

# Initialize H2O
h2o.init()

# Generate synthetic data for testing
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
df["target"] = y
h2o_df = h2o.H2OFrame(df)
h2o_df["target"] = h2o_df["target"].asfactor()

# Split into train and test
train, test = h2o_df.split_frame(ratios=[0.8])

# Run AutoML
aml = H2OAutoML(max_runtime_secs=60)  # Short test
aml.train(y="target", training_frame=train)

# Check leaderboard
lb = aml.leaderboard
print(lb.head())

# Predict
predictions = aml.predict(test)
print(predictions.head())

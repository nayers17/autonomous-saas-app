# Example: models/placeholder_model.py


class PlaceholderModel:
    def train(self, X, y):
        print("Simulating model training...")

    def predict(self, X):
        print("Simulating model prediction...")
        return [0] * len(X)  # Return a list of zeros as dummy predictions

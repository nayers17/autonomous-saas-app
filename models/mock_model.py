# models/mock_model.py


class MockModel:
    def train(self, X, y):
        print("Training mock model...")

    def predict(self, X):
        print("Predicting with mock model...")
        return [0] * len(X)  # Dummy predictions

from sklearn.linear_model import LogisticRegression
import joblib

class TraditionalModel:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def initialize(self, X_initial_vec, y_initial):
        """Train the traditional model on initial data."""
        self.model.fit(X_initial_vec, y_initial)
        joblib.dump(self.model, 'traditional_model.pkl')
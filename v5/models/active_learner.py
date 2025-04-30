from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.linear_model import LogisticRegression
import numpy as np

class ActiveLearnerModel:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.learner = None
        self.X_pool_vec = None
        self.X_pool = None
        self.y_pool = None
        self.uncertain_samples = []
        self.feedback_data = []
        self.feedback_count = 0

    def initialize(self, X_initial_vec, y_initial, X_pool_vec, X_pool, y_pool):
        """Initialize the ActiveLearner with initial training data and pool."""
        self.learner = ActiveLearner(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            query_strategy=uncertainty_sampling,
            X_training=X_initial_vec,
            y_training=y_initial
        )
        self.X_pool_vec = X_pool_vec
        self.X_pool = X_pool
        self.y_pool = y_pool

    def get_sample(self):
        """Query and return a sample for feedback."""
        if not self.uncertain_samples:
            print(f"Querying new batch of 5 samples from pool of size {self.X_pool_vec.shape[0]}")
            query_idx, _ = self.learner.query(self.X_pool_vec, n_instances=5)
            for idx in query_idx:
                text = self.X_pool[idx]
                text_vec = self.vectorizer.transform([text])
                self.uncertain_samples.append({
                    'pool_idx': int(idx),
                    'text': str(text),
                    'model_prediction': int(self.learner.predict(text_vec)[0]),
                    'model_confidence': float(np.max(self.learner.predict_proba(text_vec)))
                })
        return self.uncertain_samples.pop(0)

    def remove_sample_from_pool(self, pool_idx):
        """Remove a sample from the pool."""
        mask = np.ones(self.X_pool_vec.shape[0], dtype=bool)
        mask[pool_idx] = False
        self.X_pool_vec = self.X_pool_vec[mask]
        self.X_pool = np.delete(self.X_pool, pool_idx)
        self.y_pool = np.delete(self.y_pool, pool_idx)

    def store_feedback(self, feedback):
        """Store valid feedback data."""
        self.feedback_data.append({
            'text': str(feedback['text']),
            'model_prediction': int(feedback['model_prediction']),
            'corrected_label': int(feedback['corrected_label']),
            'model_confidence': float(feedback['model_confidence']),
            'iteration': self.feedback_count + 1
        })
        self.feedback_count += 1
        print(f"Feedback count: {self.feedback_count}, Feedback data size: {len(self.feedback_data)}")

    def retrain(self):
        """Retrain the HITL model with feedback data."""
        if self.feedback_data:
            print(f"Retraining HITL model with {len(self.feedback_data)} feedback samples")
            all_feedback_X = self.vectorizer.transform([fd['text'] for fd in self.feedback_data])
            all_feedback_y = np.array([fd['corrected_label'] for fd in self.feedback_data])
            self.learner.teach(X=all_feedback_X, y=all_feedback_y)
            print("Retraining completed successfully")
        else:
            print("No feedback data to retrain")
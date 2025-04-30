import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        """Load and preprocess the spam dataset."""
        data = pd.read_csv(self.dataset_path)
        data = data.dropna()
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        print(f"Selected dataset size: {len(data)}")
        X = data['text'].values
        y = data['label'].values
        return X, y

    def split_data(self, X, y):
        """Split data into initial training, pool, and test sets."""
        X_initial, X_pool, y_initial, y_pool = train_test_split(
            X, y, train_size=0.05, stratify=y, random_state=42
        )
        X_pool, X_test, y_pool, y_test = train_test_split(
            X_pool, y_pool, test_size=0.2, stratify=y_pool, random_state=42
        )
        print(f"Initial training set size: {len(X_initial)}")
        print(f"Unlabeled pool size: {len(X_pool)}")
        print(f"Test set size: {len(X_test)}")
        return X_initial, X_pool, y_initial, y_pool, X_test, y_test
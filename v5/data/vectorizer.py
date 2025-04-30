from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit_transform(self, X_initial, X_pool, X_test):
        """Fit vectorizer on initial data and transform all sets."""
        X_initial_vec = self.vectorizer.fit_transform(X_initial)
        X_pool_vec = self.vectorizer.transform(X_pool)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_initial_vec, X_pool_vec, X_test_vec

    def transform(self, texts):
        """Transform text data into TF-IDF features."""
        return self.vectorizer.transform(texts)
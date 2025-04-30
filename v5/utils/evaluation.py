from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a model's accuracy on a test set."""
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    print(f"{model_name} accuracy on test set (size={X_test.shape[0]}): {accuracy}")
    return accuracy
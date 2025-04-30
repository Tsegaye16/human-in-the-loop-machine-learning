from flask import Flask, request, render_template, jsonify
from models.active_learner import ActiveLearnerModel
from models.traditional_model import TraditionalModel
from data.data_loader import DataLoader
from data.vectorizer import Vectorizer
from utils.evaluation import evaluate_model
from utils.reporting import ReportGenerator
from utils.json_encoder import NumpyEncoder
import os

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = NumpyEncoder

# Initialize components
data_loader = DataLoader('spam.csv')
vectorizer = Vectorizer()
active_learner = ActiveLearnerModel(vectorizer)
traditional_model = TraditionalModel(vectorizer)
report_generator = ReportGenerator(active_learner, traditional_model, data_loader)

# Load and split data
X, y = data_loader.load_data()
X_initial, X_pool, y_initial, y_pool, X_test, y_test = data_loader.split_data(X, y)
X_initial_vec, X_pool_vec, X_test_vec = vectorizer.fit_transform(X_initial, X_pool, X_test)

# Initialize models
active_learner.initialize(X_initial_vec, y_initial, X_pool_vec, X_pool, y_pool)
traditional_model.initialize(X_initial_vec, y_initial)

# Initial evaluation
initial_hitl_accuracy = evaluate_model(active_learner.learner, X_test_vec, y_test, "HITL Initial")
initial_traditional_accuracy = evaluate_model(traditional_model.model, X_test_vec, y_test, "Traditional Initial")
report_generator.initialize_performance_history(initial_hitl_accuracy, initial_traditional_accuracy)

@app.route('/')
def home():
    return render_template('feedback.html')

@app.route('/get_sample', methods=['GET'])
def get_sample():
    try:
        sample = active_learner.get_sample()
        return jsonify(sample)
    except Exception as e:
        print(f"Error querying samples: {e}")
        return jsonify({'error': 'Failed to query samples'}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback = request.json
        pool_idx = int(feedback['pool_idx'])
        corrected_label = int(feedback['corrected_label'])
        
        # Remove sample from pool
        active_learner.remove_sample_from_pool(pool_idx)
        
        # Handle skipped feedback
        if corrected_label == -1:
            print(f"Feedback skipped for pool_idx {pool_idx}")
            return jsonify({'status': 'success', 'retrained': False})
        
        # Store feedback and retrain
        active_learner.store_feedback(feedback)
        active_learner.retrain()
        
        # Evaluate models
        hitl_accuracy = evaluate_model(active_learner.learner, X_test_vec, y_test, "HITL Post-Retrain")
        traditional_accuracy = evaluate_model(traditional_model.model, X_test_vec, y_test, "Traditional Post-Retrain")
        print(f"Post-retraining: HITL accuracy = {hitl_accuracy}, Traditional accuracy = {traditional_accuracy}")
        
        # Update performance history and generate report
        report_generator.update_performance_history(hitl_accuracy, traditional_accuracy)
        report_generator.generate_report()
        
        return jsonify({'status': 'success', 'retrained': True})
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return jsonify({'error': 'Failed to process feedback'}), 500

@app.route('/get_performance')
def get_performance():
    print(f"Returning performance history: {report_generator.performance_history}")
    return jsonify(report_generator.performance_history)

@app.route('/get_report')
def get_report():
    report = report_generator.generate_report()
    return jsonify({'report': report})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)
import matplotlib.pyplot as plt
import json

class ReportGenerator:
    def __init__(self, active_learner, traditional_model, data_loader):
        self.active_learner = active_learner
        self.traditional_model = traditional_model
        self.data_loader = data_loader
        self.performance_history = {'hitl': [], 'traditional': [], 'iterations': []}

    def initialize_performance_history(self, initial_hitl_accuracy, initial_traditional_accuracy):
        """Initialize performance history with initial accuracies."""
        self.performance_history['hitl'].append(initial_hitl_accuracy)
        self.performance_history['traditional'].append(initial_traditional_accuracy)
        self.performance_history['iterations'].append(0)

    def update_performance_history(self, hitl_accuracy, traditional_accuracy):
        """Update performance history with new accuracies."""
        self.performance_history['hitl'].append(hitl_accuracy)
        self.performance_history['traditional'].append(traditional_accuracy)
        self.performance_history['iterations'].append(self.active_learner.feedback_count)

    def generate_report(self):
        """Generate a performance report and learning curve plot."""
        from utils.evaluation import evaluate_model  # Delayed import to avoid circular dependency
        final_hitl_accuracy = evaluate_model(self.active_learner.learner, self.data_loader.X_test_vec, self.data_loader.y_test, "HITL Final")
        final_traditional_accuracy = evaluate_model(self.traditional_model.model, self.data_loader.X_test_vec, self.data_loader.y_test, "Traditional Final")
        print(f"Generating report: HITL accuracy = {final_hitl_accuracy}, Traditional accuracy = {final_traditional_accuracy}")

        plt.figure(figsize=(10, 6))
        plt.plot(self.performance_history['iterations'], self.performance_history['hitl'], label='HITL Model')
        plt.plot(self.performance_history['iterations'], self.performance_history['traditional'], label='Traditional Model')
        plt.xlabel('Feedback Samples')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve Comparison (100-Entry Dataset, Retrain Per Feedback)')
        plt.legend()
        plt.savefig('static/learning_curve.png')
        plt.close()

        report = {
            'final_hitl_accuracy': final_hitl_accuracy,
            'final_traditional_accuracy': final_traditional_accuracy,
            'improvement': final_hitl_accuracy - final_traditional_accuracy,
            'feedback_samples': len(self.active_learner.feedback_data),
            'test_set_size': self.data_loader.X_test_vec.shape[0],
            'dataset_size': len(self.data_loader.data),
            'note': 'Both models evaluated on the same test dataset with 100 total entries, HITL retrained per feedback'
        }

        with open('report.json', 'w') as f:
            json.dump(report, f, indent=2)
        return report
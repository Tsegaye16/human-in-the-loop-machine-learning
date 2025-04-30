# visualization.py
# Generates performance reports and plots for the spam classification system

import matplotlib.pyplot as plt
import json
import os

def generate_report(classifier, output_dir='static'):
    """
    Generate a performance report and learning curve plot.
    
    Args:
        classifier (SpamClassifier): Trained classifier with performance history.
        output_dir (str): Directory to save plot and report.
    
    Returns:
        dict: Report with performance metrics.
    """
    final_hitl_accuracy = classifier.evaluate_model(classifier.learner, "HITL Final")
    final_traditional_accuracy = classifier.evaluate_model(classifier.traditional_model, "Traditional Final")
    
    print(f"Generating report: HITL accuracy = {final_hitl_accuracy}, Traditional accuracy = {final_traditional_accuracy}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(classifier.performance_history['iterations'], classifier.performance_history['hitl'], label='HITL Model')
    plt.plot(classifier.performance_history['iterations'], classifier.performance_history['traditional'], label='Traditional Model')
    plt.xlabel('Feedback Samples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve Comparison')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()
    
    # Generate confidence plot
    if classifier.feedback_data:
        plt.figure(figsize=(10, 6))
        confidences = [fd['model_confidence'] for fd in classifier.feedback_data]
        iterations = [fd['iteration'] for fd in classifier.feedback_data]
        plt.plot(iterations, confidences, marker='o', label='Model Confidence')
        plt.xlabel('Feedback Samples')
        plt.ylabel('Model Confidence (Max Probability)')
        plt.title('Model Confidence Over Feedback Iterations')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'confidence_curve.png'))
        plt.close()
    
    # Create report
    report = {
        'final_hitl_accuracy': final_hitl_accuracy,
        'final_traditional_accuracy': final_traditional_accuracy,
        'improvement': final_hitl_accuracy - final_traditional_accuracy,
        'feedback_samples': len(classifier.feedback_data),
        'test_set_size': classifier.X_test_vec.shape[0],
        'dataset_size': classifier.performance_history['hitl'][0],  # Initial dataset size
        'note': 'Both models evaluated on the same test dataset, HITL retrained per feedback'
    }
    
    # Save report
    with open('report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
# Spam Active Learning System

This project implements an **active learning system** for spam email classification using a Human-in-the-Loop (HITL) approach. It compares an active learning model (HITL) with a static traditional model, demonstrating how user feedback improves classification accuracy. The system is built with Flask for user interaction, scikit-learn for machine learning, and modAL for active learning, organized in a modular Python structure.

## Table of Contents

- [Project Overview](#project-overview)
- [What is Active Learning?](#what-is-active-learning)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Interface Integration](#interface-integration)
- [How It Works](#how-it-works)
- [Expected Output](#expected-output)
- [For Developers](#for-developers)
- [License](#license)

## Project Overview

The Spam Active Learning System classifies emails as spam (1) or ham (0) using a logistic regression model. The HITL model starts with a small training set (5% of the dataset) and improves by querying uncertain samples for user feedback. A traditional model, trained on the same initial data, serves as a static baseline. The system visualizes performance via a learning curve, showing how feedback enhances the HITL model’s accuracy.

The project is modularized for clarity and maintainability, with separate modules for data loading, vectorization, model training, evaluation, reporting, and Flask routes.

## What is Active Learning?

Active learning is a machine learning approach where the model **actively selects the most informative data points** for labeling, reducing the need for large labeled datasets. In this project:

- **Problem**: Labeling all emails as spam or ham is time-consuming.
- **Solution**: The HITL model identifies emails it’s **least confident about** (e.g., with prediction probabilities near 0.5) using **uncertainty sampling**.
- **Process**:
  1. The model starts with a small labeled dataset (50 emails, 5% of 1000).
  2. It queries the user for labels on uncertain emails (e.g., “Is ‘Get free money!’ spam?”).
  3. User feedback (labels) is used to retrain the model incrementally.
  4. The model improves with minimal labeling, outperforming the static traditional model.
- **Benefit**: Achieves high accuracy (e.g., 0.89 vs. 0.85 for the traditional model) with few labeled samples (e.g., 5 feedback samples).

**Analogy**: Imagine a student (the model) studying for an exam. Instead of reviewing all questions, they ask the teacher (user) for help only on the hardest ones. This targeted learning is faster and more effective.

## Features

- **Active Learning**: Queries uncertain emails for user feedback using modAL’s `ActiveLearner` with uncertainty sampling.
- **Flask Interface**: Web-based UI (`feedback.html`) for users to view emails, see model predictions/confidence, and submit labels (ham, spam, or skip).
- **Modular Design**: Organized into modules for data loading, vectorization, models, evaluation, and reporting.
- **Performance Tracking**: Generates a learning curve (`static/learning_curve.png`) and report (`report.json`) comparing HITL and traditional model accuracies.
- **Error Handling**: Robust try-except blocks ensure the app doesn’t crash on invalid inputs or empty pools.

## Prerequisites

- Python 3.8+
- A `spam.csv` dataset with columns `text` (email content) and `label` (ham/spam).
- A `feedback.html` template in `templates/` for the Flask UI.

**Sample `spam.csv` format**:

```csv
text,label
"Get free money!",spam
"Hello friend",ham
```

---

Dependencies (listed in requirements.txt):

```
pandas
scikit-learn
modAL
matplotlib
flask
joblib
scipy
numpy
```

## Setup Instructions

Clone the Repository (or create the structure manually):

```bash
git clone https://github.com/Tsegaye16/human-in-the-loop-machine-learning
cd human-in-the-loop-machine-learning/v5
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

Prepare the Dataset:
Place your spam.csv in the project root.
Ensure it has text and label columns, with labels as ham or spam.
Prepare the UI:
Ensure feedback.html exists in templates/. A sample is provided below:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Spam Active Learning</title>
    <script>
      function fetchSample() {
        fetch("/get_sample")
          .then((response) => response.json())
          .then((data) => {
            document.getElementById("email").innerText = data.text;
            document.getElementById("prediction").innerText =
              data.model_prediction;
            document.getElementById("confidence").innerText =
              data.model_confidence.toFixed(2);
            document.getElementById("pool_idx").value = data.pool_idx;
          });
      }

      function submitFeedback() {
        const pool_idx = document.getElementById("pool_idx").value;
        const corrected_label = document.getElementById("label").value;
        fetch("/submit_feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pool_idx: pool_idx,
            corrected_label: corrected_label,
            text: document.getElementById("email").innerText,
            model_prediction: document.getElementById("prediction").innerText,
            model_confidence: document.getElementById("confidence").innerText,
          }),
        })
          .then((response) => response.json())
          .then((data) => {
            alert("Feedback submitted!");
            fetchSample();
          });
      }

      window.onload = fetchSample;
    </script>
  </head>
  <body>
    <h1>Spam Active Learning</h1>
    <p><strong>Email:</strong> <span id="email"></span></p>
    <p>
      <strong>Model Prediction (0=ham, 1=spam):</strong>
      <span id="prediction"></span>
    </p>
    <p><strong>Model Confidence:</strong> <span id="confidence"></span></p>
    <input type="hidden" id="pool_idx" />
    <label>Label:</label>
    <select id="label">
      <option value="0">Ham</option>
      <option value="1">Spam</option>
      <option value="-1">Skip</option>
    </select>
    <button onclick="submitFeedback()">Submit Feedback</button>
  </body>
</html>
```

Running the Application
Start the Flask App:

```bash
python app.py
```

---

### Detailed Explanation of Active Learning

Active learning is a semi-supervised machine learning paradigm where the model **interactively queries a human oracle** (user) to label the most informative data points, reducing labeling effort while maximizing performance. Here’s a deeper dive tailored to your project and presentation:

1. **Core Concept**:

   - **Problem**: Labeling a large dataset (e.g., 1000 emails) is costly and time-intensive.
   - **Solution**: Start with a small labeled dataset (50 emails) and let the model select which additional samples need labels based on **uncertainty**.
   - **Uncertainty Sampling**: The model picks samples where its predictions are least confident (e.g., probabilities `[0.51, 0.49]` for ham/spam), as these are most likely to improve the decision boundary.

2. **Implementation in Your Project**:

   - **Initialization** (`ActiveLearnerModel.initialize`):
     - The HITL model (`ActiveLearner`) is trained on 50 emails (5% of 1000) using logistic regression.
     - Initial accuracy: ~0.85 on the test set (190 samples).
   - **Querying** (`ActiveLearnerModel.get_sample`):
     - Uses `modAL`’s `uncertainty_sampling` to select 5 emails from the pool (760 samples) with probabilities closest to 0.5.
     - Example: An email with `[0.51, 0.49]` is queried because it’s near the decision boundary.
   - **Feedback** (`ActiveLearnerModel.store_feedback`):
     - Users label emails as ham (0), spam (1), or skip (-1) via the Flask UI.
     - Feedback is stored in `feedback_data` (e.g., `{text: "Get free money!", corrected_label: 1, ...}`).
   - **Retraining** (`ActiveLearnerModel.retrain`):
     - The model is updated incrementally with feedback using `learner.teach`, combining initial data (50 samples) and feedback (e.g., 5 samples).
     - Example: Accuracy improves from 0.85 to 0.89 after 5 feedback samples.
   - **Evaluation** (`utils/evaluation.py`):
     - Compares HITL accuracy (improving) with traditional model accuracy (static at 0.85).

3. **Why It’s Effective**:

   - **Efficiency**: Labels only the most informative samples, achieving high accuracy with minimal user effort (e.g., 5 labels vs. 760 for full pool).
   - **Human-in-the-Loop**: Leverages human expertise to correct model errors, especially on ambiguous emails.
   - **Visualization**: The learning curve (`learning_curve.png`) shows HITL accuracy rising, proving active learning’s value.

4. **Presentation Points**:

   - **Slide**: “What is Active Learning?”
     - Diagram:
       ```
       Small Labeled Data (50 emails) → Query Uncertain Samples → User Labels → Retrain → Improved Model
       ```
     - Bullet Points:
       - Starts with 5% labeled data (50 emails).
       - Queries emails with low confidence (e.g., 0.51).
       - User feedback refines the model, boosting accuracy.
       - Outperforms static model with fewer labels.
   - **Demo**: Show the UI querying an email with low confidence (e.g., 0.51), user labeling it, and the learning curve updating.

5. **Analogy** (for Audience):
   - “Active learning is like a chef perfecting a recipe. Instead of tasting every dish, they ask a critic to judge only the trickiest ones (e.g., ‘Is this too spicy?’). Each critique makes the recipe better, faster.”

---

### Detailed Explanation of Interface Integration

The Flask-based web interface (`feedback.html`) is tightly integrated with the backend to facilitate the active learning loop. Here’s a detailed breakdown for your project and presentation:

1. **Frontend (`feedback.html`)**:

   - **Purpose**: Provides an intuitive UI for users to:
     - View an email, its model prediction (0=ham, 1=spam), and confidence score (e.g., 0.51).
     - Submit a corrected label (ham, spam, or skip).
   - **Components**:
     - **Display**: Shows email text, prediction, and confidence using `<span>` elements.
     - **Input**: A `<select>` dropdown for labels (0, 1, -1) and a hidden `<input>` for `pool_idx`.
     - **Button**: A “Submit Feedback” button triggers feedback submission.
   - **JavaScript**:
     - `fetchSample()`: Calls `GET /get_sample` to retrieve a sample, updates the UI with JSON data (`text`, `model_prediction`, `model_confidence`, `pool_idx`).
     - `submitFeedback()`: Sends feedback to `POST /submit_feedback` with JSON payload, then fetches a new sample.
     - Example JSON sent to `/submit_feedback`:
       ```json
       {
         "pool_idx": 42,
         "text": "Get free money!",
         "model_prediction": 0,
         "model_confidence": 0.51,
         "corrected_label": 1
       }
       ```

2. **Backend (Routes in `app.py`)**:

   - **`/get_sample`**:
     - Calls `ActiveLearnerModel.get_sample` to query 5 uncertain samples (if `uncertain_samples` is empty) and return one as JSON.
     - Logs: “Querying new batch of 5 samples from pool of size 760”.
   - **`/submit_feedback`**:
     - Processes feedback JSON, removes the sample from the pool (`ActiveLearnerModel.remove_sample_from_pool`), stores valid feedback, retrains the model, and updates the learning curve (`ReportGenerator`).
     - Handles skips (`corrected_label = -1`) by removing the sample without retraining.
     - Logs: “Feedback count: 1, Feedback data size: 1”, “HITL accuracy = 0.86”.
   - **`/get_performance` and `/get_report`**:
     - Provide performance history and reports for potential UI enhancements (e.g., displaying the learning curve).

3. **Integration Mechanics**:

   - **Flow**:
     1. User loads `feedback.html`, triggering `fetchSample()` to call `/get_sample`.
     2. Backend returns a sample, and JavaScript populates the UI.
     3. User selects a label and clicks “Submit”, sending feedback to `/submit_feedback`.
     4. Backend processes feedback, retrains, and the UI fetches a new sample.
   - **Data Flow**:
     - Backend (`ActiveLearnerModel`) manages the pool (`X_pool_vec`, `X_pool`) and feedback (`feedback_data`).
     - Frontend sends/receives JSON, ensuring seamless communication.
   - **Error Handling**:
     - Backend uses try-except to handle invalid `pool_idx` or empty pools, returning error JSON (e.g., `{"error": "Failed to query samples"}`).
     - Frontend alerts users on successful submission.

4. **Why It’s Effective**:

   - **User-Friendly**: The UI is simple, showing only essential information (email, prediction, confidence) and a clear labeling option.
   - **Real-Time Feedback**: Asynchronous fetch calls ensure the UI updates instantly after submission.
   - **Transparency**: Displaying confidence scores (e.g., 0.51) shows why the model needs help, engaging users.
   - **Modularity**: Backend logic (`ActiveLearnerModel`, `ReportGenerator`) is decoupled from the UI, making updates easy.

5. **Presentation Points**:

   - **Slide**: “Interface Integration”
     - Diagram:
       ```
       UI (feedback.html) ↔ GET /get_sample → ActiveLearnerModel.get_sample
                           ↔ POST /submit_feedback → ActiveLearnerModel.retrain
                                                     → ReportGenerator.update
       ```
     - Bullet Points:
       - UI displays email, prediction, and confidence.
       - JavaScript fetches samples and submits feedback via JSON.
       - Backend queries uncertain samples and retrains the model.
       - Learning curve updates after each feedback, shown in `static/`.
   - **Demo**:
     - Open the UI, show an email with low confidence (e.g., 0.51), label it as spam, submit, and display the console log (“HITL accuracy = 0.86”).
     - Show `learning_curve.png` updating in `static/`.

6. **Analogy** (for Audience):
   - “The interface is like a teacher’s desk. The model (student) brings a tricky email (question) to the desk (UI). The teacher (user) marks it as spam or ham, and the student learns, updating their notes (learning curve) right away.”

---

### How to Use for Your Presentation

- **README in Slides**:

  - Include sections like “What is Active Learning?” and “Interface Integration” as slides, using the diagrams and bullet points.
  - Show the “Expected Output” section to preview console logs and `report.json`.

- **Active Learning Demo**:

  - Explain the loop: “The model queries uncertain emails, users label them, and accuracy improves.”
  - Show the UI and console logs:
    ```
    Querying new batch of 5 samples from pool of size 760
    Feedback count: 1, Feedback data size: 1
    Post-retraining: HITL accuracy = 0.86
    ```
  - Display `learning_curve.png` to show accuracy rising.

- **Interface Demo**:

  - Open `http://localhost:5000`, label an email, and show the UI updating with a new sample.
  - Highlight confidence scores (e.g., 0.51) to explain why the model needs feedback.

- **Visuals**:

  - Use the flowcharts from the README.
  - Add a confidence plot in `ReportGenerator`:
    ```python
    plt.plot([fd['iteration'] for fd in self.active_learner.feedback_data], [fd['model_confidence'] for fd in self.active_learner.feedback_data], marker='o')
    plt.savefig('static/confidence_curve.png')
    ```

- **Narrative**:
  - “Our active learning system lets users teach the model by labeling only the toughest emails. The modular backend handles querying, retraining, and tracking, while the Flask UI makes it easy and interactive, as seen in the learning curve.”

---

### Additional Notes

- **Dataset**: If you don’t have `spam.csv`, use a public dataset (e.g., Enron or SMS Spam Collection) and preprocess it to match the `text`, `label` format.
- **UI Enhancements**: Add a button to display `learning_curve.png` in the UI by serving it via Flask’s `static` route.
- **Error Handling**: The README mentions robust error handling, which you can demo by submitting invalid feedback (e.g., empty label) and showing the error response.

If you need specific slide content, a sample `spam.csv`, or further assistance with the demo setup, let me know immediately! Good luck with your presentation!

```

```

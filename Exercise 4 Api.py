from flask import Flask, request, jsonify
import joblib
import numpy as np


# Create Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict_spam():
    """
    Implement prediction endpoint
    - Accept JSON with 'message' field
    - Return prediction (spam/ham) and confidence
    """
    try:
        # Get message from request
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Preprocess and vectorize the message
        message_vectorized = vectorizer.transform([message])
        
        # Make prediction
        prediction = model.predict(message_vectorized)[0]
        
        # Get confidence score (distance from decision boundary for SVM)
        if hasattr(model, 'decision_function'):
            confidence_score = model.decision_function(message_vectorized)[0]
            # Convert to probability-like score (0-1)
            confidence = 1 / (1 + np.exp(-confidence_score))
        else:
            # For models without decision_function, use predict_proba if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(message_vectorized)[0]
                confidence = max(proba)
            else:
                confidence = 1.0  # Default confidence if no probability available
        
        # Prepare response
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(confidence),
            'message_length': len(message),
            'status': 'success'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'spam_detector'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

print("API created! Run the cell below to test it.")
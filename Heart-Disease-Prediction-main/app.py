from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])

        # Create a DataFrame for the input
        user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])

        # Make a prediction
        prediction = model.predict(user_data)
        result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease"

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

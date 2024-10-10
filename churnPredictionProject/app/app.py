from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('./models/model_rf_v1.pkl')

# Home route to show the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route (for handling form submissions)
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the HTML form
    try:
        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['monthlycharges'])
        gender_f = int(request.form['gender_f'])
        gender_m = int(request.form['gender_m'])
        contract_one_year = int(request.form['contract1'])
        contract_two_year = int(request.form['contract2'])
        
        # Ensure the input is structured as expected by the model
        features = np.array([tenure, monthly_charges, gender_f,gender_m, contract_one_year, contract_two_year])
        
        # Make prediction
        prediction = model.predict([features])[0]
        result = "Customer will churn" if prediction == 1 else "Customer will not churn"
        
        # Render the HTML template with the prediction result
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

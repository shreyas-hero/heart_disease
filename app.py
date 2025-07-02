from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved ML model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        input_data = [float(x) for x in request.form.values()]
        print(input_data)
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction using loaded model
        prediction = model.predict(input_array)
        result = "The person has heart disease." if prediction[0] == 1 else "The person does not have heart disease."
        
        # Render the result page with prediction
        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)

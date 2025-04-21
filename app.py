from flask import Flask, request, render_template
import numpy as np
import pickle


model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", message=None, input_values={})

@app.route('/predict', methods=['POST'])
def predict():
    
    radius_mean = float(request.form['radius_mean'])
    texture_mean = float(request.form['texture_mean'])
    perimeter_mean = float(request.form['perimeter_mean'])
    area_mean = float(request.form['area_mean'])  
    smoothness_mean = float(request.form['smoothness_mean'])
    compactness_mean = float(request.form['compactness_mean'])
    concavity_mean = float(request.form['concavity_mean'])
    concave_points_mean = float(request.form['concave_points_mean'])
    symmetry_mean = float(request.form['symmetry_mean'])

    
    input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, 
                            smoothness_mean, compactness_mean, concavity_mean, 
                            concave_points_mean, symmetry_mean]])

    
    if np.any(input_data < -1) or np.any(input_data > 1):
        error_message = "Input values must be between 0 and 1."
        return render_template("index.html", message=error_message, input_values={})

    
    input_data = scaler.transform(input_data)
    
    
    prediction = model.predict(input_data)

    
    output = "Cancrous" if prediction[0] == 1 else "Not Cancrous"

    
    input_values = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave_points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean
    }

    
    return render_template("index.html", message=output, input_values=input_values)
@app.route('/new_data', methods=['POST'])
def new_data():
    
    return render_template("index.html", message=None, input_values={})

# Main
if __name__ == "__main__":
    app.run(debug=True)
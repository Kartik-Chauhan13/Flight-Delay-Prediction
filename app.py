from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
with open('FlightPrediction.pkl', 'rb') as file:
    clf = pickle.load(file)

def preprocess_input(data):
    # Convert categorical variables to numerical (example)
    # This part should match the preprocessing steps used during training
    # airline_map = {"AirlineA": 0, "AirlineB": 1}  # Example mapping
    # airport_map = {"AirportX": 0, "AirportY": 1}  # Example mapping
    
    # data['airline'] = airline_map.get(data['airline'], -1)
    # data['airportFrom'] = airport_map.get(data['airportFrom'], -1)
    # data['airportTo'] = airport_map.get(data['airportTo'], -1)
    
    # Convert all values to floats (or the required type)
    for key in data:
        data[key] = float(data[key])
    
    return list(data.values())
@app.route('/')
def welcome():
    return render_template("demo.html")

@app.route("/predict", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        try:
            # Get input data from the form
            input_data = {
                "airline": request.form.get("airline"),
                "flight": request.form.get("flight"),
                "from": request.form.get("from"),
                "to": request.form.get("to"),
                "day": request.form.get("day"),
                "time": request.form.get("time"),
                "length": request.form.get("length")
            }

             #Preprocess the input data
            processed_data = preprocess_input(input_data)

            # Make a prediction using your model
            prediction = clf.predict([processed_data])[0]

            # Render the template with the prediction demo
            return render_template("result.html", prediction=prediction)
        except Exception as e:
            return render_template("result.html", error=str(e))

    else:
        # Render the form
        return render_template('demo.html')


if __name__ == "__main__":
    app.run(debug=True)

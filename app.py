from flask import Flask, render_template, request
import numpy as np
import pickle
import warnings


# Load your machine learning model
model_path = r"C:\Users\Rohith\Downloads\Online Payment Fraud Detection\flask\onlinefraudDetection.pkl"

#Load the saved model
app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

# Ignore warnings during model loading
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    model = pickle.load(open(model_path, 'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/pred", methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        # Your prediction logic here
        x = [[float(x) for x in request.form.values()]]
        print("Input Data:", x)
        x = np.array(x)
        print("Input Shape:", x.shape)

        print("Input Data Array:", x)
        pred = model.predict(x)
        print("Raw Prediction:", pred[0])

        # Map prediction to labels
        label_mapping = {0: "Not Fraud", 1: "Fraud"}
        prediction_text = label_mapping.get(pred[0], "Unknown")

        print("Final Prediction:", prediction_text)
        return render_template('submit.html', prediction_text=prediction_text)
    else:
        return render_template('submit.html', prediction_text=None)

if __name__ == "__main__":
    app.run(debug=True)


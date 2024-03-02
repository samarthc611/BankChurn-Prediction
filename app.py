from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load the models from the file
with open('model.pkl', 'rb') as file:
    loaded_models = pickle.load(file)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = []
    for key in request.form.keys():
        try:
            int_features.append(int(request.form[key]))
        except ValueError:
            pass

    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = loaded_models['LR'].predict(final)
    print(prediction)

if __name__ == '__main__':
    app.run(debug=True)

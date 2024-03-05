from flask import Flask, request, render_template
import pickle
import numpy as np
from flask import jsonify

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

    
    if int_features[-1] == 1:
        algo = 'Knn'
    if int_features[-1] == 2:
        algo = 'LR'
    if int_features[-1] == 3:
        algo = 'svc'
    if int_features[-1] == 4:
        algo = 'DTC'
    if int_features[-1] == 5:
        algo = 'rfc'
    
    int_features.pop(11)
    int_features.pop(0)


    # int_features.insert(10, int_features.pop(9))
    # int_features.insert(9, 1)
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = loaded_models[algo].predict(final)
    print(prediction)

    # return jsonify({'prediction': prediction.tolist()})
    if prediction == 0:
            pred = "Prediction: Will Not Exit"
    else:
            pred = "Prediction: Will Exit"

    return render_template("index.html", pred=pred)



if __name__ == '__main__':
    app.run(debug=True)


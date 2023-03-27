from flask import Flask, render_template, request
import pickle

from predict_disease import predict_disease

app = Flask(__name__)

with open('models/symptoms.plk', 'rb') as f:
    symptoms = pickle.load(f)

with open('models/RandomForestClassifier.plk', 'rb') as f:
    rfc = pickle.load(f)

with open('./models/precaution_dict.plk', 'rb') as f:
    precaution_dict = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)


@app.route('/predict', methods=['POST'])
def predict():
    user_input = list(request.form.values())

    # Remove empty strings if any
    while '' in user_input:
        user_input.remove('')

    return render_template('predict.html',  prediction=predict_disease(user_input, rfc), precaution=precaution_dict)



if __name__ == '__main__':
    app.run(debug=True)

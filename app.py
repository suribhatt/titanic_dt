# importing the necessary dependencies
from flask import Flask, render_template, request, jsonify
import sklearn
import pickle
import joblib

app = Flask(__name__)  # initializing a flask app


@app.route('/')  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def index():
    # # reading the inputs given by the user
    age_ = (request.form['age'])
    pclass_ = (request.form['pclass'])
    parch = (request.form['parch'])
    sibsp_ = (request.form['sibsp'])
    fare_ = (request.form['fare'])
    gender_ = (request.form['gender'])
    q_ = (request.form['q'])
    s_ = (request.form['s'])

    filename = "titanic.sav"
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    # predictions using the loaded model file
    prediction = loaded_model.predict([[pclass_, age_, sibsp_, parch, fare_, gender_, q_, s_]])
    print('prediction is', prediction)
    # showing the prediction results in a UI
    return render_template('results.html', prediction=round(100 * prediction[0]))


# @app.route('/postman', methods=['POST', 'GET'])  # route to show the predictions in a web UI
# def postm():
#
#     print(request.get_json(force=True))
#     data= request.get_json(force=True)
#     age = (data['age'])
#     parch = (data['parch'])
#     pclass = (data['pclass'])
#     sibsp = (data['sibsp'])
#     fare = (data['fare'])
#     gender = (data['gender'])
#     q = (data['q'])
#     s = (data['s'])
#
#     filename = "titanic.sav"
#     loaded_model = joblib.load(open(filename, 'rb'))  # loading the model file from the storage
#     # predictions using the loaded model file
#     prediction = loaded_model.predict([[pclass, age, sibsp, parch, fare, gender, q, s]])
#     print('prediction is', prediction)
#     prediction = round(100 * prediction[0])
#     # showing the prediction results in a UI
#     return jsonify({'Prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True)  # running the app

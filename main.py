from flask import Flask, render_template, request, redirect, url_for
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


app = Flask(__name__)
pkl_file = open('model.pkl','rb')
model = pickle.load(open('model.pkl', 'rb'))
index_dict = pickle.load(pkl_file)


dummy_credentials = {
    "abc@gmail.com": "root21",
    "xyz@yahoo.com": "shock91"
}

@app.route('/')
def goto_main():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    if email in dummy_credentials and dummy_credentials[email] == password:
        return redirect(url_for('goto_index'))
    else:
        return redirect(url_for('goto_main'))


@app.route('/index.html')
def goto_index():
    return render_template("index.html")

@app.route('/started.html')
def goto_get_started():
    return render_template("started.html")



@app.route('/predict',methods=['POST'])
def predict():

    if request.method=='POST':
        result = request.form

        index_dict = pickle.load(open('cat','rb'))
        location_cat = pickle.load(open('location_cat','rb'))

        new_vector = np.zeros(151)

        result_location = result['location']

        if result_location not in location_cat:
            new_vector[146] = 1
        else:
            new_vector[index_dict[str(result['location'])]] = 1


        new_vector[index_dict[str(result['area'])]] = 1

        new_vector[0] = result['sqft']
        new_vector[1] = result['bath']
        new_vector[2] = result['balcony']
        new_vector[3] = result['size']

    new = [new_vector]

    prediction = model.predict(new)
    rounded_prediction = round(prediction[0], 3)
    return render_template('started.html', Predict_score ='Your house estimate price is ₹ {} lakhs'.format(rounded_prediction))


if __name__ == "__main__":
    app.run(debug=True)

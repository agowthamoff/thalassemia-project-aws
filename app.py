from flask import Flask, render_template, request
from joblib import load
import random
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = load('best_knn.joblib')


@app.route('/', methods=['GET', 'POST'])
def predictor():
    if request.method == 'POST':
        # Fetch data from the form
        param1 = request.form.get('param1')
        param2 = request.form.get('param2')
        param3 = request.form.get('param3')
        param4 = request.form.get('param4')
        param5 = request.form.get('param5')
        param6 = request.form.get('param6')
        param7 = request.form.get('param7')
        param8 = request.form.get('param8')
        param9 = request.form.get('param9')
        param10 = request.form.get('param10')
        param11 = request.form.get('param11')
        param12 = request.form.get('param12')
        param13 = request.form.get('param13')
        param14 = request.form.get('param14')
        param15 = request.form.get('param15')

        # List of features expected by the model
        features_to_encode = ['hb', 'pcv', 'rbc', 'mcv', 'mch', 'mchc', 'rdw', 'wbc', 'neut', 'lymph', 'plt', 'hba',
                              'hba2', 'hbf', 'sex_encoded']

        # Prepare the input data for prediction
        input_data = [
            [param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12, param13,
             param14, param15]]
        input_df = pd.DataFrame(input_data, columns=features_to_encode)

        # Make prediction using the trained model
        prediction = model.predict(input_df)
        pred = prediction.flatten().astype(int)

        # Interpret the result
        if pred[0] == 1:
            overall = "Alpha Carrier with thalassemia"
        elif pred[0] == 0:
            overall = "Normal"

        # Doctor details (to be displayed randomly)
        doctors = [
            {"name": "Dr. John Doe", "qualification": "MD", "address": "123 Main St, City, Country"},
            {"name": "Dr. Jane Smith", "qualification": "PhD", "address": "456 Elm St, Town, Country"}
        ]

        # Randomly select a doctor
        selected_doctor = random.choice(doctors)

        # Extract doctor's information
        doctor_name = selected_doctor["name"]
        doctor_qualification = selected_doctor["qualification"]
        doctor_address = selected_doctor["address"]

        # Return the result to be displayed on the HTML page
        return render_template('main2.html', result=pred[0], overall=overall, doctor_name=doctor_name,
                               doctor_qualification=doctor_qualification, doctor_address=doctor_address)

    return render_template('main2.html')


if __name__ == '__main__':
    app.run(debug=True)

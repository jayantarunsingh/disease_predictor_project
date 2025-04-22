from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import os

# Use relative paths so it works in Docker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained models and encoder
model1 = joblib.load(os.path.join(BASE_DIR, 'model_decision_tree_gini.pkl'))
model2 = joblib.load(os.path.join(BASE_DIR, 'model_decision_tree_entropy.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

# Load symptoms list and additional info
severity_df = pd.read_csv(os.path.join(BASE_DIR, 'Symptom-severity.csv'))
desc_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_Description.csv'))
prec_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_precaution.csv'))

# Clean and prepare symptoms list
symptoms_list = sorted(severity_df['Symptom'].str.strip().str.replace(' ', '_').str.lower().unique())

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
    
    pred1 = model1.predict([input_data])[0]
    pred2 = model2.predict([input_data])[0]
    final_pred = le.inverse_transform([np.bincount([pred1, pred2]).argmax()])[0]

    description = desc_df[desc_df['Disease'].str.lower() == final_pred]['Description'].values
    precautions = prec_df[prec_df['Disease'].str.lower() == final_pred].values

    description_text = description[0] if description.size else "No description available."
    precaution_list = [p for p in precautions[0][1:] if pd.notna(p)] if precautions.size else []

    return render_template('result.html', disease=final_pred.title(), 
                           description=description_text, precautions=precaution_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # allow external access in Docker


from os import pipe

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
data = pd.read_csv(r'D:\Learn_AI_&_ML\Projects\House_price_pred\dataset\cleaned_data.csv')
pipe = pickle.load(open(r'D:\Learn_AI_&_ML\Projects\House_price_pred\model\house_price_model_grad.pkl', 'rb'))

@app.route('/')
def home():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    sqft = float(request.form.get('total_sqft'))
    bath = int(request.form.get('bath'))
    bhk = int(request.form.get('bhk')) 

    print(location, sqft, bath, bhk)
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_df)[0] * 1e5

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5000)

import pickle
import pandas as pd
import json
import os
from flask import Flask, request, Response
from health_insurance_class.health_insurance import HealthInsurance

# loading model
model = pickle.load(open('models/ML/linear_regression_model.pkl', 'rb'))

# initialize API
app = Flask(__name__)
#app.config['JSON_SORT_KEYS'] = False

@app.route('/predict', methods=['POST'])

def health_insurance_predict():
    test_json = request.get_json()
       
    test_raw = pd.json_normalize(json.loads(test_json))

    #if test_json:       # if there is data
    #    if isinstance(test_json, dict):     # verifies if test_jason is one line data
    #        test_raw = pd.DataFrame(test_json, index=[0])
    #    else:
    #        test_raw = pd.DataFrame(test_json, columns=test_json[0].keys)

    pipeline = HealthInsurance()
    df1 = pipeline.columns_rename(test_raw)
    df2 = pipeline.feature_engineering(df1)
    df3 = pipeline.data_preparation(df2)
    df_response = pipeline.get_prediction(model=model, original_data=test_raw, test_data=df3)
    return df_response
    
    #else:
    #    return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(port=port, host='0.0.0.0', debug=False)
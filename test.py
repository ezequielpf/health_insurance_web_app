import requests
import pandas as pd
import json

df_raw = pd.read_csv('/home/ezequiel/Documentos/Comunidade_DS/health_insurance_cross_sell/data/raw/train.csv')

data = json.dumps(df_raw.sample(500).to_dict(orient='records'))

# API call
#url = 'http://localhost:5000/predict'
url = 'https://health-insurance-web-app.onrender.com/predict'
#header = {'Content-type': 'application/jason'}
data = data

#r = requests.post(url, data=data, headers=header)
r = requests.post(url=url, json=data)
print(f'Status Code {r.status_code}')

d1 = pd.json_normalize(r.json()).head(20).sort_values('score', ascending=False)

print(d1['score'])
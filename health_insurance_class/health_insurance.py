import pandas as pd
import pickle
from sklearn import preprocessing, metrics
#import json

class HealthInsurance:

    def __init__(self):
        self.age_feature = pickle.load(open('models/features/age_feature.pkl', 'rb'))
        self.annual_premium_feature = pickle.load(open('models/features/annual_premium_feature.pkl', 'rb'))
        self.gender_feature = pickle.load(open('models/features/gender_feature.pkl', 'rb'))
        self.policy_sales_channel_feature = pickle.load(open('models/features/policy_sales_channel_feature.pkl', 'rb'))
        self.region_code_feature = pickle.load(open('models/features/region_code_feature.pkl', 'rb'))
        self.vehicle_age_feature = pickle.load(open('models/features/vehicle_age_feature.pkl', 'rb'))
        self.vintage_feature = pickle.load(open('models/features/vintage_feature.pkl', 'rb'))

    def columns_rename(self, data):
        cols_names = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age',
                      'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage', 'response']
        data.columns = cols_names
        return data
    
    def feature_engineering(self, data):
        data['vehicle_age'] = data['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else
                                              'between_1_2_years' if x == '1-2 Year' else
                                              'below_1_year')
        data['vehicle_damage'] = data['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        return data
    
    def data_preparation(self, data):
        data['annual_premium'] = self.annual_premium_feature.transform(data[['annual_premium']])

        data['age'] = self.age_feature.transform(data[['age']])
        
        data['vintage'] = self.vintage_feature.transform(data[['vintage']])
        
        data['gender'] = self.gender_feature.transform(data[['gender']])

        data['region_code'] = self.region_code_feature.transform(data[['region_code']])
                
        vehicle_age_encoded = self.vehicle_age_feature.transform(data[['vehicle_age']])
        vehicle_age_encoded = vehicle_age_encoded.toarray()
        aux = pd.DataFrame(vehicle_age_encoded, columns=self.vehicle_age_feature.get_feature_names_out(), index=data.index)
        data = pd.concat([data, aux], axis=1)
        data.drop('vehicle_age', axis=1, inplace=True)

        data.loc[:, 'policy_sales_channel'] = data['policy_sales_channel'].map(self.policy_sales_channel_feature)

        selected_features = ['vintage', 'annual_premium', 'region_code', 'age', 'gender', 'vehicle_damage', 'policy_sales_channel', 'previously_insured']

        return data[selected_features]
    
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict_proba(test_data)
        original_data['score'] = pred[:,1]
        return original_data.to_json(orient = 'records')
import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, save_model, load_model
df = pd.read_csv("C:/Users/johnm/intro_datascience/week 3/cleandata.csv", index_col='customerID')
df.drop('daily_charges',axis=1,inplace=True)
df.rename({'tchargeratio'  :  'charge_per_tenure'}, axis=1, inplace=True)
print(df.head())
print(df.Churn.value_counts(normalize=True))

automl = setup(df, target='Churn')
print(automl)

print(automl[2])

best_model = compare_models()
print(best_model)
print(df.iloc[-10:-1].shape)
preds_test=df.iloc[-10:-1]
print(preds_test)
print(predict_model(best_model,preds_test))
save_model(best_model, 'logistic')

loaded_logistic = load_model('logistic')
loaded_logistic

print(predict_model(loaded_logistic,preds_test))

%run data_preds_fun.py



my_preds=predict_model(best_model,df)
my_preds

p_list=[]
for i in my_preds['Score']:
    if i>=.90:
        p_list.append("90th ")
        
    elif i>=.80:
        p_list.append("80th")
        
    elif i >=.70:
       p_list.append("70th")
    elif i>=.60:
        p_list.append("60th")
        
    elif i>= .50:
        p_list.append("50th")
    elif i>=.40:
        p_list.append("40th")
        
    elif i >=.30:
        p_list.append("30th")
    elif i>= .20:
        p_list.append("20th")
        
    elif i>= .10:
        p_list.append("10th")
        
    else:
        p_list.append("low")
    
print(p_list)


my_preds["percentile"]=p_list
print(my_preds.sort_values(by="percentile",ascending=False).head(50))
9/10



def clean_data(full_df=df_fix):
    full_df['PhoneService'] =  full_df['PhoneService'].replace({'No': 0, 'Yes': 1})
    full_df['Contract'] =      full_df['Contract'].replace({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    full_df['PaymentMethod'] = full_df['PaymentMethod'].replace({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2,'Credit card (automatic)':3})
    #full_df['Churn'] =         full_df['Churn'].replace({'No': 0, 'Yes':1})

    full_df['charge_per_tenure']= full_df['TotalCharges'] / full_df['tenure']
    return full_df

    
df_fix = pd.read_csv("C:/Users/johnm/intro_datascience/week 5/new_churn_data_unmodified.csv", index_col='customerID')
print(df_fix)

print(clean_data(df_fix))
cleans=clean_data(df_fix)
def make(df):
  
    model = load_model('logistic')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions.rename({'Score': 'Churn_prob'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'},
                                            inplace=True)
    return predictions[['Churn_prob','Churn_prediction']]

make(df=cleans)


import h2o

h2o.init()

hf = h2o.H2OFrame(pd.read_csv("C:/Users/johnm/intro_datascience/churn_data.csv", index_col='customerID'))
hf
hf.types
print(hf)

from h2o.estimators import H2ORandomForestEstimator

predictors = hf.columns
predictors.remove('Churn')
response = 'Churn'

# Split the dataset into a train and valid set:
train, valid = hf.split_frame(ratios=[.8], seed=1234, )

# Build and train the model:
drf = H2ORandomForestEstimator(ntrees=50,
                                    max_depth=2,
                                    calibrate_model=True,
                                    calibration_frame=valid)
drf.train(x=predictors,
           y=response,
           training_frame=train,
           validation_frame=valid)

# Eval performance:
perf = drf.model_performance(valid=valid)
perf

drf.varimp_plot(7)
import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads Churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model('logistic')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions.rename({'Score': 'Churn_prob'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'},
                                            inplace=True)
    return predictions[['Churn_prob','Churn_prediction']]


if __name__ == "__main__":
    df = load_data("C:/Users/johnm/intro_datascience/week 5/new_churn_data.csv")
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
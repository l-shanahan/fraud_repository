from data_utils import *
from model_utils import *
from joblib import load

import warnings
warnings.filterwarnings('ignore')

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

#load the model from the file
modelname = f'{config["model_name"]}.joblib'
classifier = load(modelname)

filepath = config["running_file_path"]

df_orders, df_payment_methods, df_transactions, df_customers = read_data_from_file(filepath)
feature_matrix, user_emails = get_feature_matrix(df_orders, df_payment_methods, df_transactions, df_customers)

if 'fraudulent' in feature_matrix.columns:
    X, y = separate_xy(feature_matrix)
else:
    pass

#running the model to predict
y_pred = prediction_func(classifier, X).tolist()
y_pred_label = ['Fraudulent' if x == 1 else 'Not fraudulent' for x in y_pred]

# Assert that y_pred and user_emails are the same length
assert len(y_pred_label) == len(user_emails), "Error: y_pred and user_emails are not the same length"

# Save a JSON file with user_emails as keys and corresponding y_pred as values
email_prediction_dict = dict(zip(user_emails, y_pred_label))
with open(f'{config["output_filename"]}.json', 'w') as file:
    json.dump(email_prediction_dict, file)

print("Output JSON file has been saved.")


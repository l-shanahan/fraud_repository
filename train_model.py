from data_utils import *
from model_utils import *

from joblib import dump

import warnings
warnings.filterwarnings('ignore')

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

filepath = config["training_file_path"]

df_orders, df_payment_methods, df_transactions, df_customers = read_data_from_file(filepath)
feature_matrix, user_emails = get_feature_matrix(df_orders, df_payment_methods, df_transactions, df_customers)
classifier = create_model(feature_matrix)

modelname = f'{config["model_name"]}.joblib'
dump(classifier, modelname)
print('Model saved as', modelname)